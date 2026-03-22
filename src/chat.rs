//! RIG 기능 확장 모듈 — 커스텀 chat
//!
//! RIG의 `.prompt()` + `.with_history()` + `.extended_details()` 를 조합하여
//! &mut history 기반, usage 포함, 도구 호출 지원하는 통합 chat 함수 제공.
//!
//! ## 설계 원칙
//! - `&mut Vec<Message>` 참조: clone 없음, RIG가 자동 push
//! - RIG의 기존 Agent/Prompt 인터페이스와 호환
//! - 나중에 수동 루프(모델 분리)로 확장 가능한 기반
//!
//! ## 사용법
//! ```ignore
//! use crate::chat::{chat, chat_with_tools, ChatResult};
//!
//! // 단일 턴 (도구 없음)
//! let result = chat(&agent, "대협, 처음 뵙겠소.", &mut history).await?;
//!
//! // 도구 호출 가능
//! let result = chat_with_tools(&agent, "무당검법이 뭐요?", &mut history, 5).await?;
//! ```

use rig::agent::{Agent, PromptResponse};
use rig::client::CompletionClient;
use rig::completion::{Message, PromptError, Prompt};
use rig::providers::openai;

use bge_m3_onnx_rust::BgeM3Embedder;
use crate::rag::{VectorStore, format_context};

/// chat 결과 — 응답 텍스트 + 토큰 사용량
#[derive(Debug, Clone)]
pub struct ChatResult {
    /// LLM의 최종 응답 텍스트
    pub response: String,
    /// 토큰 사용량
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    pub cached_input_tokens: u64,
}

/// 단일 턴 chat — 도구 없는 에이전트용
///
/// history에 user + assistant 메시지가 자동 push됨.
pub async fn chat(
    agent: &Agent<openai::CompletionModel>,
    prompt: &str,
    history: &mut Vec<Message>,
) -> Result<ChatResult, PromptError> {
    let resp = agent.prompt(prompt)
        .with_history(history)
        .extended_details()
        .await?;

    Ok(to_chat_result(resp))
}

/// 도구 호출 가능 chat — max_turns 지정
///
/// RIG의 자동 도구 호출 루프를 활용.
/// history에 user + tool_call + tool_result + assistant 메시지가 자동 push됨.
pub async fn chat_with_tools(
    agent: &Agent<openai::CompletionModel>,
    prompt: &str,
    history: &mut Vec<Message>,
    max_turns: usize,
) -> Result<ChatResult, PromptError> {
    let resp = agent.prompt(prompt)
        .with_history(history)
        .max_turns(max_turns)
        .extended_details()
        .await?;

    Ok(to_chat_result(resp))
}

/// RAG 기반 chat — 벡터 검색 후 preamble에 합쳐서 응답
///
/// 1. BGE-M3로 질문을 임베딩
/// 2. 벡터 스토어에서 top_k 유사 문서 검색
/// 3. 검색 결과를 system_prompt에 합침 (<file> 태그 없이)
/// 4. agent 빌드 → chat() 호출
pub async fn chat_with_rag(
    client: &openai::CompletionsClient,
    system_prompt: &str,
    prompt: &str,
    history: &mut Vec<Message>,
    embedder: &mut BgeM3Embedder,
    store: &VectorStore,
    top_k: usize,
) -> Result<ChatResult, PromptError> {
    // 벡터 검색
    let results = store.query(prompt, embedder, top_k)
        .map_err(|e| PromptError::CompletionError(
            rig::completion::CompletionError::RequestError(e.to_string().into())
        ))?;

    // preamble에 검색 결과 합침
    let full_system = if results.is_empty() {
        system_prompt.to_string()
    } else {
        let context = format_context(&results);
        tracing::info!("[rag] {}개 문서 검색됨 (최고 유사도: {:.3})",
            results.len(), results[0].score);
        format!("{}\n\n[참고 자료]\n{}", system_prompt, context)
    };

    let agent = client.agent("local-model")
        .preamble(&full_system)
        .build();

    chat(&agent, prompt, history).await
}

/// PromptResponse → ChatResult 변환
fn to_chat_result(resp: PromptResponse) -> ChatResult {
    ChatResult {
        response: resp.output,
        input_tokens: resp.total_usage.input_tokens,
        output_tokens: resp.total_usage.output_tokens,
        total_tokens: resp.total_usage.total_tokens,
        cached_input_tokens: resp.total_usage.cached_input_tokens,
    }
}
