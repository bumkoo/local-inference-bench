use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// RIG 기능 정의 - 실험에서 사용할 RIG API 종류와 파라미터
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Feature {
    pub feature: FeatureMeta,
    /// feature.type에 따라 해당 섹션만 존재
    pub completion: Option<CompletionParams>,
    pub agent: Option<AgentParams>,
    pub agent_tool: Option<AgentToolParams>,
    pub multi_turn: Option<MultiTurnParams>,
    pub rag: Option<RagParams>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureMeta {
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// 디스패치 키: completion | agent | agent_tool | multi_turn
    #[serde(rename = "type")]
    pub feature_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionParams {
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentParams {
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentToolParams {
    #[serde(default = "default_max_tokens_large")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_max_turns")]
    pub max_turns: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiTurnParams {
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_max_turns")]
    pub max_turns: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagParams {
    #[serde(default = "default_max_tokens_large")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// 벡터 검색 상위 N개
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// ONNX 모델 경로
    #[serde(default = "default_onnx_model")]
    pub onnx_model: String,
    /// 토크나이저 경로
    #[serde(default = "default_tokenizer")]
    pub tokenizer: String,
    /// NPC 지식 데이터 경로
    #[serde(default = "default_knowledge_path")]
    pub knowledge_path: String,
}

fn default_max_tokens() -> u32 { 256 }
fn default_max_tokens_large() -> u32 { 512 }
fn default_temperature() -> f32 { 0.7 }
fn default_max_turns() -> u32 { 5 }
fn default_top_k() -> usize { 3 }
fn default_onnx_model() -> String { "../models/bge-m3/model_quantized.onnx".to_string() }
fn default_tokenizer() -> String { "../models/bge-m3/tokenizer.json".to_string() }
fn default_knowledge_path() -> String { "config/data/npc-knowledge.toml".to_string() }

impl Feature {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("기능 정의 로드 실패 {:?}: {}", path, e))?;
        let feature: Self = toml::from_str(&content)?;
        Ok(feature)
    }

    pub fn load_by_name(name: &str) -> anyhow::Result<Self> {
        let path = PathBuf::from("config/features").join(format!("{}.toml", name));
        Self::load(&path)
    }

    pub fn list_all() -> anyhow::Result<Vec<(String, Self)>> {
        let dir = PathBuf::from("config/features");
        let mut features = Vec::new();
        if dir.exists() {
            for entry in std::fs::read_dir(&dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().map_or(false, |ext| ext == "toml") {
                    if let Ok(f) = Self::load(&path) {
                        let name = path.file_stem().unwrap().to_string_lossy().to_string();
                        features.push((name, f));
                    }
                }
            }
        }
        Ok(features)
    }

    pub fn feature_type(&self) -> &str {
        &self.feature.feature_type
    }
}
