use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// 프롬프트 세트 - 하나의 실험에서 사용할 프롬프트 모음
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptSet {
    pub prompt_set: PromptSetMeta,
    #[serde(default = "default_system_prompt")]
    pub system_prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_repeat")]
    pub repeat: u32,
    pub prompts: Vec<Prompt>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptSetMeta {
    pub name: String,
    #[serde(default)]
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prompt {
    pub id: String,
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub context: String,
    pub user: String,
}

fn default_system_prompt() -> String { String::new() }
fn default_max_tokens() -> u32 { 256 }
fn default_repeat() -> u32 { 1 }

impl Prompt {
    /// system_prompt + context를 결합한 최종 시스템 프롬프트 생성
    pub fn build_system_prompt(&self, base_system: &str) -> String {
        if self.context.is_empty() {
            base_system.to_string()
        } else {
            format!("{}\n\n{}", base_system, self.context)
        }
    }
}

impl PromptSet {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("프롬프트 세트 로드 실패 {:?}: {}", path, e))?;
        let set: Self = toml::from_str(&content)?;
        Ok(set)
    }

    pub fn load_by_name(name: &str) -> anyhow::Result<Self> {
        let path = PathBuf::from("config/prompts").join(format!("{}.toml", name));
        Self::load(&path)
    }

    pub fn list_all() -> anyhow::Result<Vec<(String, Self)>> {
        let dir = PathBuf::from("config/prompts");
        let mut sets = Vec::new();
        if dir.exists() {
            for entry in std::fs::read_dir(&dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().map_or(false, |ext| ext == "toml") {
                    if let Ok(set) = Self::load(&path) {
                        let name = path.file_stem().unwrap().to_string_lossy().to_string();
                        sets.push((name, set));
                    }
                }
            }
        }
        Ok(sets)
    }

    /// repeat를 포함한 총 실행 횟수
    pub fn total_runs(&self) -> usize {
        self.prompts.len() * self.repeat as usize
    }
}
