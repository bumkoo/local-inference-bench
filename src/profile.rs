use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// 서버 프로파일 - 하나의 서버 설정 조합
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerProfile {
    pub profile: ProfileMeta,
    pub server: ServerConfig,
    pub model: ModelConfig,
    #[serde(default)]
    pub cache: CacheConfig,
    pub speculation: Option<SpeculationConfig>,
    #[serde(default)]
    pub timeouts: TimeoutsConfig,
    #[serde(default)]
    pub toolchain: ToolchainConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileMeta {
    pub name: String,
    #[serde(default)]
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default = "default_ctx_size")]
    pub ctx_size: u32,
    #[serde(default = "default_batch_size")]
    pub batch_size: u32,
    #[serde(default = "default_parallel")]
    pub parallel: u32,
    #[serde(default)]
    pub no_webui: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub main: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    #[serde(default = "default_cache_type")]
    pub type_k: String,
    #[serde(default = "default_cache_type")]
    pub type_v: String,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            type_k: "f16".to_string(),
            type_v: "f16".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculationConfig {
    pub draft: String,
    #[serde(default = "default_draft_max")]
    pub draft_max: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutsConfig {
    #[serde(default = "default_load_secs")]
    pub load_secs: u64,
    #[serde(default)]
    pub download_secs: u64,
}

impl Default for TimeoutsConfig {
    fn default() -> Self {
        Self { load_secs: 120, download_secs: 0 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolchainConfig {
    #[serde(default = "default_build_mode")]
    pub build_mode: String,
    #[serde(default)]
    pub curl_enabled: bool,
}

impl Default for ToolchainConfig {
    fn default() -> Self {
        Self {
            build_mode: "install_only".to_string(),
            curl_enabled: false,
        }
    }
}

// 기본값 함수들
fn default_host() -> String { "127.0.0.1".to_string() }
fn default_port() -> u16 { 8081 }
fn default_ctx_size() -> u32 { 4096 }
fn default_batch_size() -> u32 { 512 }
fn default_parallel() -> u32 { 2 }
fn default_cache_type() -> String { "f16".to_string() }
fn default_draft_max() -> u32 { 24 }
fn default_load_secs() -> u64 { 120 }
fn default_build_mode() -> String { "install_only".to_string() }

impl ServerProfile {
    /// 프로파일 TOML 파일 로드
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("프로파일 로드 실패 {:?}: {}", path, e))?;
        let profile: Self = toml::from_str(&content)?;
        profile.validate()?;
        Ok(profile)
    }

    /// profiles 디렉토리에서 이름으로 찾기
    pub fn load_by_name(name: &str) -> anyhow::Result<Self> {
        let path = PathBuf::from("config/profiles").join(format!("{}.toml", name));
        Self::load(&path)
    }

    /// 모든 프로파일 목록 조회
    pub fn list_all() -> anyhow::Result<Vec<(String, Self)>> {
        let dir = PathBuf::from("config/profiles");
        let mut profiles = Vec::new();
        if dir.exists() {
            for entry in std::fs::read_dir(&dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().map_or(false, |ext| ext == "toml") {
                    if let Ok(profile) = Self::load(&path) {
                        let name = path.file_stem().unwrap().to_string_lossy().to_string();
                        profiles.push((name, profile));
                    }
                }
            }
        }
        Ok(profiles)
    }

    fn validate(&self) -> anyhow::Result<()> {
        let main = Path::new(&self.model.main);
        if !main.exists() {
            anyhow::bail!("메인 모델 파일 없음: {:?}", main);
        }
        if let Some(ref spec) = self.speculation {
            let draft = Path::new(&spec.draft);
            if !draft.exists() {
                anyhow::bail!("드래프트 모델 파일 없음: {:?}", draft);
            }
        }
        Ok(())
    }

    pub fn is_speculative(&self) -> bool {
        self.speculation.is_some()
    }

    /// 서버 API base URL
    pub fn api_base_url(&self) -> String {
        format!("http://{}:{}/v1", self.server.host, self.server.port)
    }
}
