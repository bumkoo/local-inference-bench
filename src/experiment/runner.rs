use chrono::Utc;
use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers::openai;
use std::time::Instant;
use tracing::info;

use crate::feature::Feature;
use crate::profile::ServerProfile;
use super::prompt_set::PromptSet;
use super::store::{ExperimentStore, RunRecord};

/// 실험 러너 - 3축(Profile × Feature × PromptSet)에 따라 실행
pub struct ExperimentRunner {
    profile: ServerProfile,
    feature: Feature,
    prompt_set: PromptSet,
}

impl ExperimentRunner {
    pub fn new(profile: ServerProfile, feature: Feature, prompt_set: PromptSet) -> Self {
        Self { profile, feature, prompt_set }
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

    /// Feature: agent_tool - 도구 호출 테스트
    async fn run_agent_tool(
        &self,
        client: &openai::CompletionsClient,
        store: &mut ExperimentStore,
    ) -> anyhow::Result<()> {
        info!("  (agent_tool: 도구 호출은 추후 구현, 현재 agent 모드로 실행)");
        self.run_completion(client, store).await
    }

    /// Feature: multi_turn - 프롬프트를 순차 대화로 실행
    async fn run_multi_turn(
        &self,
        client: &openai::CompletionsClient,
        store: &mut ExperimentStore,
    ) -> anyhow::Result<()> {
        let params = self.feature.multi_turn.as_ref()
            .ok_or_else(|| anyhow::anyhow!("multi_turn 설정 없음"))?;

        let system = &self.prompt_set.system_prompt;
        let prompts = &self.prompt_set.prompts;
        let max_turns = params.max_turns.min(prompts.len() as u32) as usize;

        for repeat_idx in 0..self.prompt_set.repeat {
            info!("멀티턴 세션 {} (최대 {}턴)", repeat_idx + 1, max_turns);
            let mut conversation_context = String::new();

            for (turn, prompt) in prompts.iter().take(max_turns).enumerate() {
                let run_id = (repeat_idx as usize * max_turns) + turn + 1;
                let full_system = if conversation_context.is_empty() {
                    prompt.build_system_prompt(system)
                } else {
                    format!("{}\n\n[이전 대화]\n{}", prompt.build_system_prompt(system), conversation_context)
                };

                let agent = client.agent("local-model").preamble(&full_system).build();

                let start = Instant::now();
                let result = agent.prompt(&prompt.user).await;
                let elapsed = start.elapsed();
                let latency_ms = elapsed.as_millis() as u64;

                let (response, success) = match result {
                    Ok(text) => (text, true),
                    Err(e) => (format!("[ERROR] {}", e), false),
                };

                let tokens = estimate_tokens(&response);
                let tps = calc_tps(tokens, latency_ms);
                info!("  턴 {}: {}ms, ~{}tok", turn + 1, latency_ms, tokens);

                conversation_context.push_str(&format!("Player: {}\nNPC: {}\n", prompt.user, response));

                let extra = serde_json::json!({
                    "turn": turn + 1,
                    "total_turns": max_turns,
                    "context_length": conversation_context.len(),
                });

                store.append_run(&RunRecord {
                    run_id, prompt_id: prompt.id.clone(), prompt_label: prompt.label.clone(),
                    repeat_index: repeat_idx, timestamp: Utc::now(),
                    system_prompt: full_system, user_prompt: prompt.user.clone(),
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
