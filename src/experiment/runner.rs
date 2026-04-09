use chrono::Utc;
use rig::client::CompletionClient;
use rig::completion::Message;
use rig::providers::openai;
use std::time::Instant;
use tracing::info;

use crate::chat::{chat, chat_with_tools, chat_with_rag};
use crate::feature::Feature;
use crate::llama_client::{LlamaTimings, LlamaTimingsClient, LlamaCompletionsClient, TimingsStore};
use crate::profile::ServerProfile;
use crate::rag;
use crate::tools::{LookupKungFu, CheckRelationship, CheckInventory};
use super::prompt_set::PromptSet;
use super::store::{ExperimentStore, RunRecord};

/// 실험 러너 - 3축(Profile × Feature × PromptSet)에 따라 실행
pub struct ExperimentRunner {
    profile: ServerProfile,
    feature: Feature,
    prompt_set: PromptSet,
    /// 실제 생성된 서버 로그 파일 경로 (타임스탬프 포함)
    server_log_file: Option<String>,
}

impl ExperimentRunner {
    pub fn new(profile: ServerProfile, feature: Feature, prompt_set: PromptSet) -> Self {
        Self { profile, feature, prompt_set, server_log_file: None }
    }

    pub fn with_server_log(mut self, log_file: Option<String>) -> Self {
        self.server_log_file = log_file;
        self
    }

    pub async fn run(&self) -> anyhow::Result<ExperimentStore> {
        let mut store = ExperimentStore::new(&self.profile, &self.feature, &self.prompt_set)?;
        info!("실험 시작: {}", store.meta.id);
        info!("  프로파일: {}", self.profile.profile.name);
        info!("  기능: {} ({})", self.feature.feature.name, self.feature.feature_type());
        info!("  프롬프트: {} ({}개 × {}회)",
            self.prompt_set.prompt_set.name,
            self.prompt_set.prompts.len(),
            self.prompt_set.repeat,
        );

        // RIG OpenAI 클라이언트 → Chat Completions API 사용
        // LlamaTimingsClient로 HTTP 응답에서 llama-server timings 캡처
        let (http_client, timings_store) = LlamaTimingsClient::new();
        let client: LlamaCompletionsClient = openai::Client::<reqwest::Client>::builder()
            .api_key("local-no-key")
            .base_url(&self.profile.api_base_url())
            .http_client(http_client)
            .build()?
            .completions_api();

        match self.feature.feature_type() {
            "completion" => self.run_completion(&client, &mut store, &timings_store).await?,
            "agent" => self.run_agent(&client, &mut store, &timings_store).await?,
            "agent_tool" => self.run_agent_tool(&client, &mut store, &timings_store).await?,
            "multi_turn" => self.run_multi_turn(&client, &mut store, &timings_store).await?,
            "rag" => self.run_rag(&client, &mut store, &timings_store).await?,
            other => anyhow::bail!("알 수 없는 feature type: {}", other),
        }

        store.finalize()?;

        // 서버 로그 필터링 (DEBUG 제거 → 실험 폴더에 server_filtered.log)
        if let Some(ref lf) = self.server_log_file {
            let log_path = std::path::Path::new(lf);
            let _ = ExperimentStore::filter_server_log(log_path, &store.dir);
        }

        if let Some(ref s) = store.meta.summary {
            info!("완료: {}/{} 성공, 평균 {:.0}ms, {:.1}tok/s",
                s.successful_runs, s.total_runs, s.avg_latency_ms, s.avg_tokens_per_sec);
        }

        Ok(store)
    }

    /// Feature: completion - 가장 단순한 프롬프트/응답
    /// chat() 사용 — history는 프롬프트별 독립 (단일턴)
    async fn run_completion(
        &self,
        client: &LlamaCompletionsClient,
        store: &mut ExperimentStore,
        timings_store: &TimingsStore,
    ) -> anyhow::Result<()> {
        let total = self.prompt_set.total_runs();
        let mut run_id = 0;

        for prompt in &self.prompt_set.prompts {
            for repeat_idx in 0..self.prompt_set.repeat {
                run_id += 1;
                info!("[{}/{}] {}", run_id, total, prompt.label);

                let system = prompt.build_system_prompt(&self.prompt_set.system_prompt);
                let agent = client.agent("local-model").preamble(&system).build();
                let mut history: Vec<Message> = Vec::new();

                let start = Instant::now();
                let result = chat(&agent, &prompt.user, &mut history).await;
                let latency_ms = start.elapsed().as_millis() as u64;
                let timings_json = collect_timings(timings_store);

                let (response, success) = match result {
                    Ok(cr) => {
                        let tps = calc_tps(cr.output_tokens as usize, latency_ms);
                        log_result(latency_ms, cr.output_tokens as usize, tps, true);
                        (cr.response, true)
                    }
                    Err(e) => {
                        info!("  에러: {}", e);
                        (format!("[ERROR] {}", e), false)
                    }
                };

                let tokens = estimate_tokens(&response);
                let tps = calc_tps(tokens, latency_ms);

                let extra = timings_json.map(|t| serde_json::json!({"timings": t}));

                store.append_run(&RunRecord {
                    run_id, prompt_id: prompt.id.clone(), prompt_label: prompt.label.clone(),
                    repeat_index: repeat_idx, timestamp: Utc::now(),
                    system_prompt: system, user_prompt: prompt.user.clone(),
                    response, success, latency_ms, tokens_generated: tokens,
                    tokens_per_sec: tps, extra,
                })?;
            }
        }
        Ok(())
    }

    /// Feature: agent - preamble 기반 에이전트
    async fn run_agent(
        &self,
        client: &LlamaCompletionsClient,
        store: &mut ExperimentStore,
        timings_store: &TimingsStore,
    ) -> anyhow::Result<()> {
        self.run_completion(client, store, timings_store).await
    }

    /// Feature: agent_tool - NPC 도구 호출 (무공 조회, 관계 확인, 소지품 확인)
    /// chat.rs의 chat_with_tools() 사용 — usage 실측, history 자동 관리
    async fn run_agent_tool(
        &self,
        client: &LlamaCompletionsClient,
        store: &mut ExperimentStore,
        timings_store: &TimingsStore,
    ) -> anyhow::Result<()> {
        let params = self.feature.agent_tool.as_ref()
            .ok_or_else(|| anyhow::anyhow!("agent_tool 설정 없음"))?;

        let total = self.prompt_set.total_runs();
        let mut run_id = 0;
        let max_turns = params.max_turns as usize;

        for prompt in &self.prompt_set.prompts {
            for repeat_idx in 0..self.prompt_set.repeat {
                run_id += 1;
                info!("[{}/{}] {} (max_turns={})", run_id, total, prompt.label, max_turns);

                let system = prompt.build_system_prompt(&self.prompt_set.system_prompt);
                let agent = client.agent("local-model")
                    .preamble(&system)
                    .tool(LookupKungFu)
                    .tool(CheckRelationship)
                    .tool(CheckInventory)
                    .max_tokens(params.max_tokens as u64)
                    .build();

                // 프롬프트별 독립 history (도구 호출 이력은 RIG가 자동 관리)
                let mut history: Vec<Message> = Vec::new();

                let start = Instant::now();
                let result = chat_with_tools(&agent, &prompt.user, &mut history, max_turns).await;
                let latency_ms = start.elapsed().as_millis() as u64;
                let timings_json = collect_timings(timings_store);

                let (response, success, mut extra) = match result {
                    Ok(cr) => {
                        let tps = calc_tps(cr.output_tokens as usize, latency_ms);
                        log_result(latency_ms, cr.output_tokens as usize, tps, true);
                        let extra = serde_json::json!({
                            "max_turns": max_turns,
                            "tools": ["lookup_kung_fu", "check_relationship", "check_inventory"],
                            "usage": {
                                "input_tokens": cr.input_tokens,
                                "output_tokens": cr.output_tokens,
                                "total_tokens": cr.total_tokens,
                                "cached_input_tokens": cr.cached_input_tokens,
                            },
                            "history_len": history.len(),
                        });
                        (cr.response, true, extra)
                    }
                    Err(e) => {
                        let err_str = format!("{}", e);
                        info!("  에러: {}", e);
                        let extra = serde_json::json!({
                            "error_type": if err_str.contains("MaxTurns") { "max_turns_exceeded" } else { "other" },
                            "error": err_str,
                        });
                        (format!("[ERROR] {}", e), false, extra)
                    }
                };

                if let Some(t) = timings_json {
                    extra.as_object_mut().unwrap().insert("timings".to_string(), t);
                }

                let tokens = estimate_tokens(&response);
                let tps = calc_tps(tokens, latency_ms);

                store.append_run(&RunRecord {
                    run_id, prompt_id: prompt.id.clone(), prompt_label: prompt.label.clone(),
                    repeat_index: repeat_idx, timestamp: Utc::now(),
                    system_prompt: system.clone(), user_prompt: prompt.user.clone(),
                    response, success, latency_ms, tokens_generated: tokens,
                    tokens_per_sec: tps, extra: Some(extra),
                })?;
            }
        }
        Ok(())
    }

    /// Feature: multi_turn - chat()을 활용한 멀티턴 대화
    /// history에 user/assistant 자동 누적, usage 실측
    async fn run_multi_turn(
        &self,
        client: &LlamaCompletionsClient,
        store: &mut ExperimentStore,
        timings_store: &TimingsStore,
    ) -> anyhow::Result<()> {
        let params = self.feature.multi_turn.as_ref()
            .ok_or_else(|| anyhow::anyhow!("multi_turn 설정 없음"))?;

        let prompts = &self.prompt_set.prompts;
        let max_turns = params.max_turns.min(prompts.len() as u32) as usize;

        for repeat_idx in 0..self.prompt_set.repeat {
            info!("멀티턴 세션 {} (최대 {}턴)", repeat_idx + 1, max_turns);

            let system = prompts[0].build_system_prompt(&self.prompt_set.system_prompt);
            let agent = client.agent("local-model").preamble(&system).build();

            let mut history: Vec<Message> = Vec::new();
            let mut conversation_log: Vec<serde_json::Value> = Vec::new();

            for (turn, prompt) in prompts.iter().take(max_turns).enumerate() {
                let run_id = (repeat_idx as usize * max_turns) + turn + 1;

                let start = Instant::now();
                let result = chat(&agent, &prompt.user, &mut history).await;
                let latency_ms = start.elapsed().as_millis() as u64;
                let timings_json = collect_timings(timings_store);

                let (response, success, usage_extra) = match result {
                    Ok(cr) => {
                        let tps = calc_tps(cr.output_tokens as usize, latency_ms);
                        info!("  턴 {}: {}ms, {}tok(in:{}/out:{}), {:.1}tok/s",
                            turn + 1, latency_ms, cr.total_tokens, cr.input_tokens, cr.output_tokens, tps);
                        let usage = serde_json::json!({
                            "input_tokens": cr.input_tokens,
                            "output_tokens": cr.output_tokens,
                            "total_tokens": cr.total_tokens,
                            "cached_input_tokens": cr.cached_input_tokens,
                        });
                        (cr.response, true, usage)
                    }
                    Err(e) => {
                        info!("  턴 {} 에러: {}", turn + 1, e);
                        (format!("[ERROR] {}", e), false, serde_json::json!(null))
                    }
                };

                conversation_log.push(serde_json::json!({
                    "role": "user", "content": prompt.user
                }));
                conversation_log.push(serde_json::json!({
                    "role": "assistant", "content": response
                }));

                let tokens = estimate_tokens(&response);
                let tps = calc_tps(tokens, latency_ms);

                let mut extra = serde_json::json!({
                    "turn": turn + 1,
                    "total_turns": max_turns,
                    "usage": usage_extra,
                    "conversation": conversation_log.clone(),
                });

                if let Some(t) = timings_json {
                    extra.as_object_mut().unwrap().insert("timings".to_string(), t);
                }

                store.append_run(&RunRecord {
                    run_id, prompt_id: prompt.id.clone(), prompt_label: prompt.label.clone(),
                    repeat_index: repeat_idx, timestamp: Utc::now(),
                    system_prompt: system.clone(), user_prompt: prompt.user.clone(),
                    response, success, latency_ms, tokens_generated: tokens,
                    tokens_per_sec: tps, extra: Some(extra),
                })?;
            }
        }
        Ok(())
    }

    /// Feature: rag - BGE-M3 벡터 검색 → preamble 합침 → LLM 응답
    async fn run_rag(
        &self,
        client: &LlamaCompletionsClient,
        store: &mut ExperimentStore,
        timings_store: &TimingsStore,
    ) -> anyhow::Result<()> {
        let params = self.feature.rag.as_ref()
            .ok_or_else(|| anyhow::anyhow!("rag 설정 없음"))?;
        let strategy = &params.strategy;

        // ort 초기화 + BGE-M3 임베더 로딩
        bge_m3_onnx_rust::init_ort();
        info!("BGE-M3 임베더 로딩: {}", params.onnx_model);
        let mut embedder = bge_m3_onnx_rust::BgeM3Embedder::new(
            &params.onnx_model, &params.tokenizer,
        )?;

        // 지식 데이터 → 벡터 스토어 적재
        let knowledge_path = std::path::Path::new(&params.knowledge_path);
        let vector_store = rag::load_knowledge(knowledge_path, &mut embedder)?;
        info!("벡터 스토어: {}개 문서 적재 완료, 전략: {}", vector_store.len(), strategy.name());

        let total = self.prompt_set.total_runs();
        let mut run_id = 0;

        for prompt in &self.prompt_set.prompts {
            for repeat_idx in 0..self.prompt_set.repeat {
                run_id += 1;
                info!("[{}/{}] {} (전략={}, top_k={})",
                    run_id, total, prompt.label, strategy.name(), strategy.top_k());

                let system = prompt.build_system_prompt(&self.prompt_set.system_prompt);
                let mut history: Vec<Message> = Vec::new();

                let start = Instant::now();

                // 전략 기반 벡터 검색 → preamble 합침 → LLM 호출
                let result = chat_with_rag(
                    client, &system, &prompt.user, &mut history,
                    &mut embedder, &vector_store, strategy,
                ).await;
                let latency_ms = start.elapsed().as_millis() as u64;
                let timings_json = collect_timings(timings_store);

                // 검색 결과도 기록 (디버깅용)
                let search_results = vector_store.query(&prompt.user, &mut embedder, strategy)
                    .ok()
                    .map(|results| results.iter().map(|r| {
                        let preview: String = r.text.chars().take(80).collect();
                        serde_json::json!({
                            "id": r.id,
                            "score": r.score,
                            "detail": r.detail,
                            "text_preview": preview,
                        })
                    }).collect::<Vec<_>>());

                let (response, success, usage_extra) = match result {
                    Ok(cr) => {
                        let tps = calc_tps(cr.output_tokens as usize, latency_ms);
                        log_result(latency_ms, cr.output_tokens as usize, tps, true);
                        let usage = serde_json::json!({
                            "input_tokens": cr.input_tokens,
                            "output_tokens": cr.output_tokens,
                            "total_tokens": cr.total_tokens,
                            "cached_input_tokens": cr.cached_input_tokens,
                        });
                        (cr.response, true, usage)
                    }
                    Err(e) => {
                        info!("  에러: {}", e);
                        (format!("[ERROR] {}", e), false, serde_json::json!(null))
                    }
                };

                let tokens = estimate_tokens(&response);
                let tps = calc_tps(tokens, latency_ms);

                let mut extra = serde_json::json!({
                    "strategy": strategy.name(),
                    "top_k": strategy.top_k(),
                    "usage": usage_extra,
                    "search_results": search_results,
                    "knowledge_docs": vector_store.len(),
                });

                if let Some(t) = timings_json {
                    extra.as_object_mut().unwrap().insert("timings".to_string(), t);
                }

                store.append_run(&RunRecord {
                    run_id, prompt_id: prompt.id.clone(), prompt_label: prompt.label.clone(),
                    repeat_index: repeat_idx, timestamp: Utc::now(),
                    system_prompt: system.clone(), user_prompt: prompt.user.clone(),
                    response, success, latency_ms, tokens_generated: tokens,
                    tokens_per_sec: tps, extra: Some(extra),
                })?;
            }
        }
        Ok(())
    }
}

/// TimingsStore에서 모든 timings를 drain하여 JSON으로 변환.
/// 단일 호출: LlamaTimings 직접 반환.
/// 다중 호출 (tool-calling): response (마지막) + tool_rounds (나머지) 분리.
fn collect_timings(store: &TimingsStore) -> Option<serde_json::Value> {
    let mut timings: Vec<LlamaTimings> = store.lock().ok()?.drain(..).collect();
    if timings.is_empty() {
        return None;
    }
    if timings.len() == 1 {
        return serde_json::to_value(timings.remove(0)).ok();
    }
    // 다중 호출: 마지막 = 최종 대사, 나머지 = 도구 호출 라운드
    let response = timings.pop().unwrap();
    let total_prompt_ms: f64 = timings.iter().map(|t| t.prompt_ms).sum::<f64>() + response.prompt_ms;
    let total_predicted_ms: f64 = timings.iter().map(|t| t.predicted_ms).sum::<f64>() + response.predicted_ms;
    Some(serde_json::json!({
        "response": response,
        "tool_rounds": timings,
        "total_prompt_ms": total_prompt_ms,
        "total_predicted_ms": total_predicted_ms,
    }))
}

fn estimate_tokens(text: &str) -> usize {
    let cjk = text.chars().filter(|c| {
        matches!(*c as u32, 0x4E00..=0x9FFF | 0xAC00..=0xD7AF | 0x3040..=0x30FF)
    }).count();
    let ascii_words = text.split_whitespace()
        .filter(|w| w.chars().all(|c| c.is_ascii()))
        .count();
    cjk + (ascii_words as f64 * 1.3) as usize
}

fn calc_tps(tokens: usize, latency_ms: u64) -> f64 {
    if latency_ms > 0 { tokens as f64 / (latency_ms as f64 / 1000.0) } else { 0.0 }
}

fn log_result(latency_ms: u64, tokens: usize, tps: f64, success: bool) {
    if success {
        info!("  완료: {}ms, ~{}tok, {:.1}tok/s", latency_ms, tokens, tps);
    } else {
        info!("  실패: {}ms", latency_ms);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    fn make_timings(prompt_n: u64, predicted_n: u64, prompt_ms: f64, predicted_ms: f64) -> LlamaTimings {
        LlamaTimings {
            prompt_n, prompt_ms,
            prompt_per_token_ms: if prompt_n > 0 { prompt_ms / prompt_n as f64 } else { 0.0 },
            prompt_per_second: if prompt_ms > 0.0 { prompt_n as f64 / (prompt_ms / 1000.0) } else { 0.0 },
            predicted_n, predicted_ms,
            predicted_per_token_ms: if predicted_n > 0 { predicted_ms / predicted_n as f64 } else { 0.0 },
            predicted_per_second: if predicted_ms > 0.0 { predicted_n as f64 / (predicted_ms / 1000.0) } else { 0.0 },
        }
    }

    fn make_store(items: Vec<LlamaTimings>) -> TimingsStore {
        Arc::new(Mutex::new(VecDeque::from(items)))
    }

    // ----------------------------------------------------------------
    // collect_timings
    // ----------------------------------------------------------------
    #[test]
    fn test_collect_timings_empty_store() {
        let store = make_store(vec![]);
        assert!(collect_timings(&store).is_none());
    }

    #[test]
    fn test_collect_timings_single_call() {
        let t = make_timings(512, 45, 89.2, 1234.5);
        let store = make_store(vec![t]);
        let result = collect_timings(&store).unwrap();

        // 단일 호출: LlamaTimings 직접 반환 (response/tool_rounds 분리 없음)
        assert_eq!(result["prompt_n"], 512);
        assert_eq!(result["predicted_n"], 45);
        assert!(result.get("response").is_none());
        assert!(result.get("tool_rounds").is_none());
    }

    #[test]
    fn test_collect_timings_multi_call_tool_calling() {
        // tool-calling: 2회 호출 (도구 1회 + 최종 대사 1회)
        let tool_round = make_timings(300, 20, 50.0, 600.0);
        let final_response = make_timings(500, 45, 80.0, 1200.0);
        let store = make_store(vec![tool_round, final_response]);

        let result = collect_timings(&store).unwrap();

        // 마지막 = response, 나머지 = tool_rounds
        assert_eq!(result["response"]["prompt_n"], 500);
        assert_eq!(result["response"]["predicted_n"], 45);
        assert_eq!(result["tool_rounds"].as_array().unwrap().len(), 1);
        assert_eq!(result["tool_rounds"][0]["prompt_n"], 300);

        // 합산 검증
        let total_prompt_ms = result["total_prompt_ms"].as_f64().unwrap();
        assert!((total_prompt_ms - 130.0).abs() < 0.01); // 50.0 + 80.0
        let total_predicted_ms = result["total_predicted_ms"].as_f64().unwrap();
        assert!((total_predicted_ms - 1800.0).abs() < 0.01); // 600.0 + 1200.0
    }

    #[test]
    fn test_collect_timings_three_calls() {
        // tool-calling 2회 + 최종 대사 1회
        let t1 = make_timings(200, 15, 30.0, 400.0);
        let t2 = make_timings(350, 25, 55.0, 700.0);
        let t3 = make_timings(500, 50, 85.0, 1400.0);
        let store = make_store(vec![t1, t2, t3]);

        let result = collect_timings(&store).unwrap();

        assert_eq!(result["response"]["predicted_n"], 50);
        assert_eq!(result["tool_rounds"].as_array().unwrap().len(), 2);
        assert_eq!(result["tool_rounds"][0]["predicted_n"], 15);
        assert_eq!(result["tool_rounds"][1]["predicted_n"], 25);

        let total_prompt_ms = result["total_prompt_ms"].as_f64().unwrap();
        assert!((total_prompt_ms - 170.0).abs() < 0.01); // 30 + 55 + 85
    }

    #[test]
    fn test_collect_timings_drains_store() {
        let store = make_store(vec![
            make_timings(100, 10, 10.0, 100.0),
            make_timings(200, 20, 20.0, 200.0),
        ]);

        // 첫 drain
        let _ = collect_timings(&store);
        assert!(store.lock().unwrap().is_empty());

        // 두 번째 호출: 빈 store → None
        assert!(collect_timings(&store).is_none());
    }

    // ----------------------------------------------------------------
    // estimate_tokens
    // ----------------------------------------------------------------
    #[test]
    fn test_estimate_tokens_ascii_only() {
        // "hello world" → 2 ASCII words × 1.3 = 2
        assert_eq!(estimate_tokens("hello world"), 2);
    }

    #[test]
    fn test_estimate_tokens_cjk_only() {
        // "안녕하세요" → 5 한글 글자 (CJK 범위)
        assert_eq!(estimate_tokens("안녕하세요"), 5);
    }

    #[test]
    fn test_estimate_tokens_mixed() {
        // "안녕 hello" → CJK 2자 + ASCII 1단어 × 1.3 = 2 + 1 = 3
        assert_eq!(estimate_tokens("안녕 hello"), 3);
    }

    #[test]
    fn test_estimate_tokens_empty() {
        assert_eq!(estimate_tokens(""), 0);
    }

    // ----------------------------------------------------------------
    // calc_tps
    // ----------------------------------------------------------------
    #[test]
    fn test_calc_tps_normal() {
        // 100 tokens / 2000ms = 50 tok/s
        let tps = calc_tps(100, 2000);
        assert!((tps - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_calc_tps_zero_latency() {
        assert_eq!(calc_tps(100, 0), 0.0);
    }

    #[test]
    fn test_calc_tps_zero_tokens() {
        assert_eq!(calc_tps(0, 1000), 0.0);
    }
}
