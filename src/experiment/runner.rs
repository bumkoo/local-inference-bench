use chrono::Utc;
use rig::client::CompletionClient;
use rig::completion::{Message, Prompt};
use rig::providers::openai;
use std::time::Instant;
use tracing::info;

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

                let start = Instant::now();
                let result = agent.prompt(&prompt.user).await;
                let elapsed = start.elapsed();
                let latency_ms = elapsed.as_millis() as u64;

                let (response, success) = match result {
                    Ok(text) => (text, true),
                    Err(e) => {
                        info!("  에러: {}", e);
                        (format!("[ERROR] {}", e), false)
                    }
                };

                let tokens = estimate_tokens(&response);
                let tps = calc_tps(tokens, latency_ms);
                log_result(latency_ms, tokens, tps, success);

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

                let start = Instant::now();
                let result = agent.prompt(&prompt.user)
                    .max_turns(max_turns)
                    .await;
                let elapsed = start.elapsed();
                let latency_ms = elapsed.as_millis() as u64;

                let (response, success, extra_info) = match result {
                    Ok(text) => (text, true, None),
                    Err(e) => {
                        // MaxTurnsError는 도구 호출이 max_turns를 초과한 경우
                        let err_str = format!("{}", e);
                        let extra = serde_json::json!({
                            "error_type": if err_str.contains("MaxTurns") { "max_turns_exceeded" } else { "other" },
                            "error": err_str,
                        });
                        info!("  에러: {}", e);
                        (format!("[ERROR] {}", e), false, Some(extra))
                    }
                };

                let tokens = estimate_tokens(&response);
                let tps = calc_tps(tokens, latency_ms);
                log_result(latency_ms, tokens, tps, success);

                let extra = extra_info.unwrap_or_else(|| serde_json::json!({
                    "max_turns": max_turns,
                    "tools": ["lookup_kung_fu", "check_relationship", "check_inventory"],
                }));

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

    /// Feature: multi_turn - RIG .with_history()를 활용한 멀티턴 대화
    /// 이전 방식: system prompt에 대화 이력을 텍스트로 합쳐서 전송
    /// 현재 방식: RIG가 messages 배열에 user/assistant를 교대로 누적
    ///   → chat template이 역할별 특수 토큰을 정확히 적용
    ///   → llama-server cache_reuse가 최대 효과 (이전 턴 전체가 접두사)
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

            // 첫 프롬프트의 context로 에이전트 생성 (동일 캐릭터와 연속 대화)
            let system = prompts[0].build_system_prompt(&self.prompt_set.system_prompt);
            let agent = client.agent("local-model").preamble(&system).build();

            // RIG가 관리하는 대화 이력 (user/assistant 메시지 자동 누적)
            let mut history: Vec<Message> = Vec::new();
            // 증빙용 간소화 대화 로그
            let mut conversation_log: Vec<serde_json::Value> = Vec::new();

            for (turn, prompt) in prompts.iter().take(max_turns).enumerate() {
                let run_id = (repeat_idx as usize * max_turns) + turn + 1;

                let start = Instant::now();
                let result = agent.prompt(&prompt.user)
                    .with_history(&mut history)
                    .await;
                let elapsed = start.elapsed();
                let latency_ms = elapsed.as_millis() as u64;

                let (response, success) = match result {
                    Ok(text) => (text, true),
                    Err(e) => {
                        info!("  턴 {} 에러: {}", turn + 1, e);
                        (format!("[ERROR] {}", e), false)
                    }
                };

                let tokens = estimate_tokens(&response);
                let tps = calc_tps(tokens, latency_ms);
                info!("  턴 {}: {}ms, ~{}tok, {:.1}tok/s", turn + 1, latency_ms, tokens, tps);

                // 증빙용 대화 로그 누적
                conversation_log.push(serde_json::json!({
                    "role": "user", "content": prompt.user
                }));
                conversation_log.push(serde_json::json!({
                    "role": "assistant", "content": response
                }));

                let extra = serde_json::json!({
                    "turn": turn + 1,
                    "total_turns": max_turns,
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
