use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use tracing::info;

use crate::profile::ServerProfile;

// ── ServerHandle ─────────────────────────────────────────

pub struct ServerHandle {
    child: Child,
    pub log_file: Option<String>,
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        info!("llama-server 종료 (PID {})", self.child.id());
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

// ── launch_server ────────────────────────────────────────

pub fn launch_server(profile: &ServerProfile) -> anyhow::Result<ServerHandle> {
    log_profile(profile);

    let binary = resolve_binary(&profile.toolchain)?;
    let main_model = std::fs::canonicalize(&profile.model.main)?;
    let log_file = resolve_log_file(profile.debug.log_file.as_deref());
    let cli_args = build_cli_args(profile, &main_model, log_file.as_deref());

    info!("llama-server: {:?}", binary);

    let child = spawn_server(&binary, &cli_args)?;
    let timeout = Duration::from_secs(profile.timeouts.load_secs);
    wait_for_health(&profile.server.host, profile.server.port, timeout)?;

    info!("서버 준비 완료: {}", profile.api_base_url());
    Ok(ServerHandle { child, log_file })
}

// ── 바이너리 경로 해석 ──────────────────────────────────

fn resolve_binary(
    toolchain: &crate::profile::ToolchainConfig,
) -> anyhow::Result<PathBuf> {
    // binary_path 직접 지정 시 우선
    if let Some(ref p) = toolchain.binary_path {
        let path = PathBuf::from(p);
        anyhow::ensure!(path.exists(), "binary_path 없음: {:?}", path);
        return Ok(path);
    }

    let tag = toolchain
        .repo_tag
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("toolchain.repo_tag 또는 binary_path 필요"))?;

    // %APPDATA%/llama_cpp_toolchain/llama_cpp/data/llama_cpp_{tag}_{backend}/bin/llama-server.exe
    let app_data = std::env::var("APPDATA")
        .map_err(|_| anyhow::anyhow!("APPDATA 환경변수 없음"))?;
    let base = PathBuf::from(&app_data)
        .join("llama_cpp_toolchain")
        .join("llama_cpp")
        .join("data");

    let prefix = format!("llama_cpp_{}_", tag);
    let entry = std::fs::read_dir(&base)?
        .filter_map(|e| e.ok())
        .find(|e| {
            let name = e.file_name();
            let s = name.to_string_lossy();
            s.starts_with(&prefix) && !s.ends_with(".lock")
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "태그 '{}' 툴체인 없음. 설치: lmcpp-toolchain-cli install --repo-tag {}",
                tag,
                tag
            )
        })?;

    let binary = entry.path().join("bin").join("llama-server.exe");
    anyhow::ensure!(binary.exists(), "llama-server.exe 없음: {:?}", binary);
    Ok(binary)
}

// ── CLI 인자 빌드 ───────────────────────────────────────

fn build_cli_args(
    profile: &ServerProfile,
    main_model: &Path,
    log_file: Option<&str>,
) -> Vec<String> {
    let mut a = Vec::new();

    // 필수
    push(&mut a, "--model", &main_model.to_string_lossy());
    push(&mut a, "--host", &profile.server.host);
    push(&mut a, "--port", &profile.server.port.to_string());

    // [server]
    if profile.server.no_webui {
        a.push("--no-webui".into());
    }
    push_opt(&mut a, "--parallel", &profile.server.parallel);
    push_opt(&mut a, "--timeout", &profile.server.timeout);

    // [gpu]
    push_opt(&mut a, "--gpu-layers", &profile.gpu.layers);
    // b8637+: --flash-attn [on|off|auto]
    if profile.gpu.flash_attn {
        push(&mut a, "--flash-attn", "on");
    }
    push_opt_str(&mut a, "--device", &profile.gpu.device);
    push_opt(&mut a, "--main-gpu", &profile.gpu.main_gpu);

    // [cpu]
    push_opt(&mut a, "--threads", &profile.cpu.threads);
    push_opt(&mut a, "--threads-batch", &profile.cpu.threads_batch);
    if profile.cpu.mlock {
        a.push("--mlock".into());
    }
    if profile.cpu.no_mmap {
        a.push("--no-mmap".into());
    }

    // [context]
    push_opt(&mut a, "--ctx-size", &profile.context.ctx_size);
    push_opt(&mut a, "--batch-size", &profile.context.batch_size);
    push_opt(&mut a, "--ubatch-size", &profile.context.ubatch_size);

    // [cache]
    push_opt_str(&mut a, "--cache-type-k", &profile.cache.type_k);
    push_opt_str(&mut a, "--cache-type-v", &profile.cache.type_v);
    push_opt(&mut a, "--cache-reuse", &profile.cache.cache_reuse);
    if profile.cache.no_kv_offload {
        a.push("--no-kv-offload".into());
    }

    // [speculation]
    if let Some(ref spec) = profile.speculation {
        if let Ok(draft) = std::fs::canonicalize(&spec.draft) {
            push(&mut a, "--model-draft", &draft.to_string_lossy());
        }
        push_opt(&mut a, "--draft-max", &spec.draft_max);
        push_opt(&mut a, "--draft-min", &spec.draft_min);
        if let Some(v) = spec.draft_p_min {
            push(&mut a, "--draft-p-min", &v.to_string());
        }
        push_opt(&mut a, "--gpu-layers-draft", &spec.gpu_layers_draft);
        push_opt(&mut a, "--ctx-size-draft", &spec.ctx_size_draft);
    }

    // [inference]
    push_opt_str(
        &mut a,
        "--reasoning-format",
        &profile.inference.reasoning_format,
    );
    if profile.inference.jinja {
        a.push("--jinja".into());
    }
    push_opt_str(&mut a, "--chat-template", &profile.inference.chat_template);
    if profile.inference.special_tokens {
        a.push("--special-tokens".into());
    }

    // [debug]
    // b8637+: --verbose-prompt, --log-prefix, --log-timestamps 제거됨
    if profile.debug.verbose {
        a.push("--verbose".into());
    }
    push_opt(&mut a, "-lv", &profile.debug.verbosity);

    // 로그 파일
    if let Some(lf) = log_file {
        push(&mut a, "--log-file", lf);
        info!("  log_file: {}", lf);
    }

    a
}

fn push(args: &mut Vec<String>, flag: &str, val: &str) {
    args.push(flag.into());
    args.push(val.into());
}

fn push_opt<T: std::fmt::Display>(args: &mut Vec<String>, flag: &str, val: &Option<T>) {
    if let Some(v) = val {
        push(args, flag, &v.to_string());
    }
}

fn push_opt_str(args: &mut Vec<String>, flag: &str, val: &Option<String>) {
    if let Some(v) = val {
        push(args, flag, v);
    }
}

// ── 로그 파일 경로 (타임스탬프 삽입) ────────────────────

fn resolve_log_file(raw: Option<&str>) -> Option<String> {
    let raw = raw?;
    let path = Path::new(raw);
    let stem = path.file_stem().unwrap_or_default().to_string_lossy();
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_string())
        .unwrap_or_default();
    let ts = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let new_name = if ext.is_empty() {
        format!("{}_{}", stem, ts)
    } else {
        format!("{}_{}.{}", stem, ts, ext)
    };
    let new_path = path.with_file_name(new_name);

    // 상대 경로 → 절대 경로 (llama-server가 다른 cwd에서 실행될 수 있음)
    let abs = if new_path.is_relative() {
        std::env::current_dir()
            .map(|cwd| cwd.join(&new_path).to_string_lossy().to_string())
            .unwrap_or_else(|_| new_path.to_string_lossy().to_string())
    } else {
        new_path.to_string_lossy().to_string()
    };

    // 로그 디렉토리 생성
    if let Some(parent) = Path::new(&abs).parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    Some(abs)
}

// ── 프로세스 실행 ───────────────────────────────────────

fn spawn_server(binary: &Path, cli_args: &[String]) -> anyhow::Result<Child> {
    let child = Command::new(binary)
        .args(cli_args)
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| anyhow::anyhow!("llama-server 실행 실패: {}", e))?;

    info!("llama-server PID {}", child.id());
    Ok(child)
}

// ── Health check ────────────────────────────────────────

fn wait_for_health(host: &str, port: u16, timeout: Duration) -> anyhow::Result<()> {
    let url = format!("http://{}:{}/health", host, port);
    let start = Instant::now();
    let poll = Duration::from_millis(500);

    loop {
        if start.elapsed() > timeout {
            anyhow::bail!(
                "llama-server health check 타임아웃 ({:.0}초)",
                timeout.as_secs_f64()
            );
        }

        match ureq::get(&url).timeout(Duration::from_secs(2)).call() {
            Ok(resp) => {
                let body = resp.into_string()?;
                if body.contains("\"ok\"") {
                    info!(
                        "llama-server healthy ({:.1}초)",
                        start.elapsed().as_secs_f64()
                    );
                    return Ok(());
                }
                // "loading model" 등 — 계속 폴링
            }
            Err(_) => {
                // ConnectionRefused 등 — 서버 시작 중, 계속 폴링
            }
        }

        std::thread::sleep(poll);
    }
}

// ── 프로필 로그 ─────────────────────────────────────────

fn log_profile(profile: &ServerProfile) {
    info!(
        "서버 시작: {} (llama.cpp {})",
        profile.profile.name,
        profile
            .toolchain
            .repo_tag
            .as_deref()
            .unwrap_or("custom binary")
    );
    info!("  모델: {}", profile.model.main);
    if let Some(ngl) = profile.gpu.layers {
        info!(
            "  GPU layers: {}{}",
            ngl,
            if profile.gpu.flash_attn {
                " +flash_attn"
            } else {
                ""
            }
        );
    }
    let ctx = profile
        .context
        .ctx_size
        .map_or("default".into(), |v| v.to_string());
    let batch = profile
        .context
        .batch_size
        .map_or("default".into(), |v| v.to_string());
    info!("  ctx: {}, batch: {}", ctx, batch);
    let ck = profile.cache.type_k.as_deref().unwrap_or("default");
    let cv = profile.cache.type_v.as_deref().unwrap_or("default");
    info!("  cache: K={}, V={}", ck, cv);
    if let Some(ref spec) = profile.speculation {
        info!(
            "  드래프트: {} (max {})",
            spec.draft,
            spec.draft_max.unwrap_or(16)
        );
    }
}
