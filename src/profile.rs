use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// 서버 프로파일 - 하나의 서버 설정 조합
/// 모든 섹션은 [profile], [model] 외에는 Option 또는 Default
/// TOML에 안 적으면 llama-server 기본값 사용
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerProfile {
    pub profile: ProfileMeta,
    pub server: ServerConfig,
    pub model: ModelConfig,
    #[serde(default)]
    pub gpu: GpuConfig,
    #[serde(default)]
    pub cpu: CpuConfig,
    #[serde(default)]
    pub context: ContextConfig,
    #[serde(default)]
    pub cache: CacheConfig,
    pub speculation: Option<SpeculationConfig>,
    #[serde(default)]
    pub inference: InferenceConfig,
    #[serde(default)]
    pub debug: DebugConfig,
    #[serde(default)]
    pub timeouts: TimeoutsConfig,
    #[serde(default)]
    pub toolchain: ToolchainConfig,
}

// ── [profile] ──────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileMeta {
    pub name: String,
    #[serde(default)]
    pub description: String,
}

// ── [server] ───────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    pub parallel: Option<u64>,
    #[serde(default)]
    pub no_webui: bool,
    pub timeout: Option<u64>,
}

// ── [model] ────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub main: String,
}

// ── [gpu] ──────────────────────────────────────────────

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GpuConfig {
    pub layers: Option<u64>,
    #[serde(default)]
    pub flash_attn: bool,
    pub device: Option<String>,
    pub main_gpu: Option<u32>,
    pub split_mode: Option<String>,
}

// ── [cpu] ──────────────────────────────────────────────

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CpuConfig {
    pub threads: Option<u64>,
    pub threads_batch: Option<u64>,
    #[serde(default)]
    pub mlock: bool,
    #[serde(default)]
    pub no_mmap: bool,
    pub numa: Option<String>,
}

// ── [context] ──────────────────────────────────────────

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContextConfig {
    pub ctx_size: Option<u64>,
    pub batch_size: Option<u64>,
    pub ubatch_size: Option<u64>,
}

// ── [cache] ────────────────────────────────────────────

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheConfig {
    pub type_k: Option<String>,
    pub type_v: Option<String>,
    pub cache_reuse: Option<u64>,
    #[serde(default)]
    pub no_kv_offload: bool,
}

// ── [speculation] ──────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculationConfig {
    pub draft: String,
    pub draft_max: Option<u64>,
    pub draft_min: Option<u64>,
    pub draft_p_min: Option<f32>,
    pub gpu_layers_draft: Option<u32>,
    pub ctx_size_draft: Option<u64>,
}

// ── [inference] ────────────────────────────────────────

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InferenceConfig {
    pub reasoning_format: Option<String>,
    #[serde(default)]
    pub jinja: bool,
    pub chat_template: Option<String>,
    #[serde(default)]
    pub special_tokens: bool,
}

// ── [debug] ────────────────────────────────────────────

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DebugConfig {
    #[serde(default)]
    pub verbose_prompt: bool,
    #[serde(default)]
    pub verbose: bool,
    pub log_file: Option<String>,
    /// llama-server 콘솔 로그 레벨 (-lv N)
    /// 0=콘솔 I만(가장 깔끔), 미설정=기본, 1+=점점 상세
    /// 주의: --log-file은 -lv와 무관하게 항상 전 레벨 기록
    pub verbosity: Option<u8>,
}

// ── [timeouts] ─────────────────────────────────────────

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

// ── [toolchain] ────────────────────────────────────────

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

// ── 기본값 ─────────────────────────────────────────────

fn default_host() -> String { "127.0.0.1".to_string() }
fn default_port() -> u16 { 8081 }
fn default_load_secs() -> u64 { 120 }
fn default_build_mode() -> String { "install_only".to_string() }

// ── impl ServerProfile ────────────────────────────────

impl ServerProfile {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("프로파일 로드 실패 {:?}: {}", path, e))?;
        let profile: Self = toml::from_str(&content)?;
        profile.validate()?;
        Ok(profile)
    }

    pub fn load_by_name(name: &str) -> anyhow::Result<Self> {
        let path = PathBuf::from("config/profiles").join(format!("{}.toml", name));
        Self::load(&path)
    }

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

    pub fn api_base_url(&self) -> String {
        format!("http://{}:{}/v1", self.server.host, self.server.port)
    }
}
