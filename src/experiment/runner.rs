use chrono::Utc;
use rig::client::CompletionClient;
use rig::completion::Message;
use rig::providers::openai;
use std::time::Instant;
use tracing::info;

use crate::chat::{chat, chat_with_tools};
use crate::feature::Feature;
use crate::profile::ServerProfile;
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
        // (기본 openai::Client는 /v1/responses를 쓰므로, .completions_api()로 전환)
        let client = openai::Client::builder()
            .api_key("local-no-key")
            .base_url(&self.profile.api_base_url())
            .build()?
            .completions_api();

        match self.feature.feature_type() {
            "completion" => self.run_completion(&client, &mut store).await?,
            "agent" => self.run_agent(&client, &mut store).await?,
            "agent_tool" => self.run_agent_tool(&client, &mut store).await?,
            "multi_turn" => self.run_multi_turn(&client, &mut store).await?,
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
        client: &openai::CompletionsClient,
        store: &mut ExperimentStore,
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

                store.append_run(&RunRecord {
                    run_id, prompt_id: prompt.id.clone(), prompt_label: prompt.label.clone(),
                    repeat_index: repeat_idx, timestamp: Utc::now(),
                    system_prompt: system, user_prompt: prompt.user.clone(),
                    response, success, latency_ms, tokens_generated: tokens,
                    tokens_per_sec: tps, extra: None,
                })?;
            }
        }
        Ok(())
    }

    /// Feature: agent - preamble 기반 에이전트
    async fn run_agent(
        &self,
        client: &openai::CompletionsClient,
        store: &mut ExperimentStore,
    ) -> anyhow::Result<()> {
        self.run_completion(client, store).await
    }

    /// Feature: agent_tool - NPC 도구 호출 (무공 조회, 관계 확인, 소지품 확인)
    /// chat.rs의 chat_with_tools() 사용 — usage 실측, history 자동 관리
    async fn run_agent_tool(
        &self,
        client: &openai::CompletionsClient,
        store: &mut ExperimentStore,
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

                let (response, success, extra) = match result {
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
        client: &openai::CompletionsClient,
        store: &mut ExperimentStore,
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

                let extra = serde_json::json!({
                    "turn": turn + 1,
                    "total_turns": max_turns,
                    "usage": usage_extra,
                    "conversation": conversation_log.clone(),
                });

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
