//! llama-server 전용 HTTP 클라이언트 래퍼
//!
//! RIG의 `HttpClientExt` trait을 구현하여, llama-server 응답의 `timings` 객체를
//! RIG 파이프라인에 영향 없이 캡처한다.
//!
//! ## 동작 원리
//! `LazyBody` future 내부에서 응답 bytes를 가로채 `timings` JSON을 파싱하고,
//! `TimingsStore`(Arc<Mutex<VecDeque>>)에 push한 뒤 원본 bytes를 그대로 RIG에 전달.

use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use rig::http_client::{
    self, HttpClientExt, LazyBody, MultipartForm, StreamingResponse,
};
use rig::providers::openai;
use rig::wasm_compat::WasmCompatSendStream;
use http::{Request, Response};
use serde::{Deserialize, Serialize};

// ================================================================
// 타입 에일리어스 — chat.rs, runner.rs에서 사용
// ================================================================
pub type LlamaCompletionModel = openai::completion::CompletionModel<LlamaTimingsClient>;
pub type LlamaCompletionsClient = openai::CompletionsClient<LlamaTimingsClient>;

// ================================================================
// LlamaTimings — llama-server /v1/chat/completions 응답의 timings 객체
// ================================================================
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaTimings {
    /// 프롬프트 평가 토큰 수
    #[serde(default)]
    pub prompt_n: u64,
    /// 프롬프트 평가 시간 (ms)
    #[serde(default)]
    pub prompt_ms: f64,
    /// 프롬프트 토큰당 시간 (ms)
    #[serde(default)]
    pub prompt_per_token_ms: f64,
    /// 프롬프트 처리 속도 (tokens/sec)
    #[serde(default)]
    pub prompt_per_second: f64,
    /// 생성(예측) 토큰 수
    #[serde(default)]
    pub predicted_n: u64,
    /// 생성 시간 (ms)
    #[serde(default)]
    pub predicted_ms: f64,
    /// 생성 토큰당 시간 (ms)
    #[serde(default)]
    pub predicted_per_token_ms: f64,
    /// 생성 속도 (tokens/sec)
    #[serde(default)]
    pub predicted_per_second: f64,
}

/// 실험 단위로 공유되는 timings 큐.
/// 각 HTTP 응답마다 push, chat 호출 후 drain.
pub type TimingsStore = Arc<Mutex<VecDeque<LlamaTimings>>>;

// ================================================================
// LlamaTimingsClient — reqwest::Client 래퍼 + timings 인터셉트
// ================================================================
#[derive(Clone, Debug)]
pub struct LlamaTimingsClient {
    inner: reqwest::Client,
    store: TimingsStore,
}

impl LlamaTimingsClient {
    /// 새 클라이언트 생성. TimingsStore 핸들도 함께 반환.
    pub fn new() -> (Self, TimingsStore) {
        let store: TimingsStore = Arc::new(Mutex::new(VecDeque::new()));
        let client = Self {
            inner: reqwest::Client::new(),
            store: store.clone(),
        };
        (client, store)
    }
}

/// `Default` 구현 — RIG 내부에서 `H::default()` 호출 시 필요.
impl Default for LlamaTimingsClient {
    fn default() -> Self {
        Self {
            inner: reqwest::Client::default(),
            store: Arc::new(Mutex::new(VecDeque::new())),
        }
    }
}

/// 응답 bytes에서 timings 파싱 → store에 push
fn extract_timings(bytes: &[u8], store: &TimingsStore) {
    if let Ok(text) = std::str::from_utf8(bytes)
        && let Ok(json) = serde_json::from_str::<serde_json::Value>(text)
        && let Some(timings_val) = json.get("timings")
        && let Ok(t) = serde_json::from_value::<LlamaTimings>(timings_val.clone())
        && let Ok(mut guard) = store.lock()
    {
        guard.push_back(t);
    }
}

// ================================================================
// HttpClientExt 구현 — reqwest::Client 구현을 미러링하되
// send()의 LazyBody 내부에서 timings 인터셉트 추가
// ================================================================
impl HttpClientExt for LlamaTimingsClient {
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + Send + 'static
    where
        T: Into<Bytes> + Send,
        U: From<Bytes> + Send + 'static,
    {
        let (parts, body) = req.into_parts();
        let body_bytes: Bytes = body.into();
        let req = self
            .inner
            .request(parts.method, parts.uri.to_string())
            .headers(parts.headers)
            .body(body_bytes);
        let store = self.store.clone();

        async move {
            let response: reqwest::Response = req.send().await
                .map_err(|e| http_client::Error::Instance(Box::new(e)))?;

            if !response.status().is_success() {
                let status = response.status();
                let text = response.text().await.unwrap_or_default();
                return Err(http_client::Error::InvalidStatusCodeWithMessage(status, text));
            }

            let mut res = Response::builder().status(response.status());
            if let Some(hs) = res.headers_mut() {
                *hs = response.headers().clone();
            }

            // LazyBody: RIG가 .into_body().await 할 때 비로소 실행됨
            let body: LazyBody<U> = Box::pin(async move {
                let bytes: Bytes = response
                    .bytes()
                    .await
                    .map_err(|e| http_client::Error::Instance(Box::new(e)))?;

                // --- timings 인터셉트 ---
                extract_timings(&bytes, &store);
                // --- 인터셉트 끝 ---

                Ok(U::from(bytes))
            });

            res.body(body).map_err(http_client::Error::Protocol)
        }
    }

    fn send_multipart<U>(
        &self,
        req: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + Send + 'static
    where
        U: From<Bytes> + Send + 'static,
    {
        let (parts, body) = req.into_parts();
        let body = reqwest::multipart::Form::from(body);

        let req = self
            .inner
            .request(parts.method, parts.uri.to_string())
            .headers(parts.headers)
            .multipart(body);

        async move {
            let response: reqwest::Response = req.send().await
                .map_err(|e| http_client::Error::Instance(Box::new(e)))?;

            if !response.status().is_success() {
                let status = response.status();
                let text = response.text().await.unwrap_or_default();
                return Err(http_client::Error::InvalidStatusCodeWithMessage(status, text));
            }

            let mut res = Response::builder().status(response.status());
            if let Some(hs) = res.headers_mut() {
                *hs = response.headers().clone();
            }

            let body: LazyBody<U> = Box::pin(async {
                let bytes: Bytes = response
                    .bytes()
                    .await
                    .map_err(|e| http_client::Error::Instance(Box::new(e)))?;
                Ok(U::from(bytes))
            });

            res.body(body).map_err(http_client::Error::Protocol)
        }
    }

    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + Send
    where
        T: Into<Bytes>,
    {
        let (parts, body) = req.into_parts();
        let body_bytes: Bytes = body.into();

        let req = self
            .inner
            .request(parts.method, parts.uri.to_string())
            .headers(parts.headers)
            .body(body_bytes)
            .build()
            .map_err(|x| http_client::Error::Instance(Box::new(x)))
            .unwrap();

        let client = self.inner.clone();

        async move {
            let response: reqwest::Response = client.execute(req).await
                .map_err(|e| http_client::Error::Instance(Box::new(e)))?;

            if !response.status().is_success() {
                let status = response.status();
                let text = response.text().await.unwrap_or_default();
                return Err(http_client::Error::InvalidStatusCodeWithMessage(status, text));
            }

            let mut res = Response::builder()
                .status(response.status())
                .version(response.version());

            if let Some(hs) = res.headers_mut() {
                *hs = response.headers().clone();
            }

            use futures::StreamExt;

            let mapped_stream: Pin<Box<dyn WasmCompatSendStream<InnerItem = http_client::Result<Bytes>>>> =
                Box::pin(
                    response
                        .bytes_stream()
                        .map(|chunk: Result<Bytes, reqwest::Error>| {
                            chunk.map_err(|e| http_client::Error::Instance(Box::new(e)))
                        }),
                );

            res.body(mapped_stream).map_err(http_client::Error::Protocol)
        }
    }
}
