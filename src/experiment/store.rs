use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::PathBuf;

use crate::feature::Feature;
use crate::profile::ServerProfile;
use super::prompt_set::PromptSet;

/// 실험 메타데이터 (experiment.json)
/// 3축: ServerProfile × Feature × PromptSet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentMeta {
    pub id: String,
    pub created_at: DateTime<Utc>,
    /// 3축 키
    pub profile_name: String,
    pub feature_name: String,
    pub prompt_set_name: String,
    /// 각 축의 스냅샷 (재현성 보장)
    pub profile_snapshot: ServerProfile,
    pub feature_snapshot: Feature,
    pub prompt_set_snapshot: PromptSet,
    /// 실험 완료 후 채워짐
    pub summary: Option<ExperimentSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentSummary {
    pub total_runs: usize,
    pub successful_runs: usize,
    pub failed_runs: usize,
    pub avg_latency_ms: f64,
    pub min_latency_ms: f64,
    pub max_latency_ms: f64,
    pub avg_tokens_per_sec: f64,
    pub avg_tokens_generated: f64,
}

/// 개별 실행 결과 (runs.jsonl 한 줄)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRecord {
    pub run_id: usize,
    pub prompt_id: String,
    pub prompt_label: String,
    pub repeat_index: u32,
    pub timestamp: DateTime<Utc>,
    pub system_prompt: String,
    pub user_prompt: String,
    pub response: String,
    pub success: bool,
    pub latency_ms: u64,
    pub tokens_generated: usize,
    pub tokens_per_sec: f64,
    /// feature-specific 메타 (tool 호출 결과, 턴 수 등)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<serde_json::Value>,
}

pub struct ExperimentStore {
    pub dir: PathBuf,
    pub meta: ExperimentMeta,
}

impl ExperimentStore {
    /// 새 실험 저장소 생성 (3축 조합)
    pub fn new(
        profile: &ServerProfile,
        feature: &Feature,
        prompt_set: &PromptSet,
    ) -> anyhow::Result<Self> {
        let now = Utc::now();
        // 디렉토리명: timestamp_profile_feature_promptset
        let id = format!(
            "{}_{}_{}_{}", 
            now.format("%Y%m%d_%H%M%S"),
            profile.profile.name,
            feature.feature.name,
            prompt_set.prompt_set.name,
        );

        let dir = PathBuf::from("data/experiments").join(&id);
        fs::create_dir_all(&dir)?;

        let meta = ExperimentMeta {
            id,
            created_at: now,
            profile_name: profile.profile.name.clone(),
            feature_name: feature.feature.name.clone(),
            prompt_set_name: prompt_set.prompt_set.name.clone(),
            profile_snapshot: profile.clone(),
            feature_snapshot: feature.clone(),
            prompt_set_snapshot: prompt_set.clone(),
            summary: None,
        };

        let json = serde_json::to_string_pretty(&meta)?;
        fs::write(dir.join("experiment.json"), json)?;

        Ok(Self { dir, meta })
    }

    pub fn append_run(&self, record: &RunRecord) -> anyhow::Result<()> {
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.dir.join("runs.jsonl"))?;
        writeln!(file, "{}", serde_json::to_string(record)?)?;
        Ok(())
    }

    pub fn finalize(&mut self) -> anyhow::Result<()> {
        let runs = self.load_runs()?;
        if runs.is_empty() { return Ok(()); }

        let success: Vec<&RunRecord> = runs.iter().filter(|r| r.success).collect();
        let total = runs.len();

        let summary = if success.is_empty() {
            ExperimentSummary {
                total_runs: total,
                successful_runs: 0,
                failed_runs: total,
                avg_latency_ms: 0.0, min_latency_ms: 0.0, max_latency_ms: 0.0,
                avg_tokens_per_sec: 0.0, avg_tokens_generated: 0.0,
            }
        } else {
            let n = success.len() as f64;
            let lats: Vec<f64> = success.iter().map(|r| r.latency_ms as f64).collect();
            ExperimentSummary {
                total_runs: total,
                successful_runs: success.len(),
                failed_runs: total - success.len(),
                avg_latency_ms: lats.iter().sum::<f64>() / n,
                min_latency_ms: lats.iter().cloned().fold(f64::INFINITY, f64::min),
                max_latency_ms: lats.iter().cloned().fold(0.0_f64, f64::max),
                avg_tokens_per_sec: success.iter().map(|r| r.tokens_per_sec).sum::<f64>() / n,
                avg_tokens_generated: success.iter().map(|r| r.tokens_generated as f64).sum::<f64>() / n,
            }
        };

        self.meta.summary = Some(summary);
        let json = serde_json::to_string_pretty(&self.meta)?;
        fs::write(self.dir.join("experiment.json"), json)?;
        Ok(())
    }

    pub fn load_runs(&self) -> anyhow::Result<Vec<RunRecord>> {
        let path = self.dir.join("runs.jsonl");
        if !path.exists() { return Ok(Vec::new()); }
        let content = fs::read_to_string(&path)?;
        Ok(content.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l))
            .collect::<Result<Vec<_>, _>>()?)
    }

    /// 모든 실험 목록 (필터링 가능)
    pub fn list_experiments(
        profile_filter: Option<&str>,
        feature_filter: Option<&str>,
        prompt_set_filter: Option<&str>,
    ) -> anyhow::Result<Vec<ExperimentMeta>> {
        let base = PathBuf::from("data/experiments");
        let mut results = Vec::new();
        if !base.exists() { return Ok(results); }
        for entry in fs::read_dir(&base)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() { continue; }
            let meta_path = entry.path().join("experiment.json");
            if !meta_path.exists() { continue; }
            let content = fs::read_to_string(&meta_path)?;
            if let Ok(meta) = serde_json::from_str::<ExperimentMeta>(&content) {
                // 부분 일치 필터 (contains)
                if let Some(pf) = profile_filter {
                    if !meta.profile_name.contains(pf) { continue; }
                }
                if let Some(ff) = feature_filter {
                    if !meta.feature_name.contains(ff) { continue; }
                }
                if let Some(ps) = prompt_set_filter {
                    if !meta.prompt_set_name.contains(ps) { continue; }
                }
                results.push(meta);
            }
        }
        results.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(results)
    }
}
