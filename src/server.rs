use lmcpp::{
    LmcppBuildInstallMode, LmcppServer, LmcppServerLauncher, LmcppToolChain,
    ReasoningFormat, ServerArgs,
};
use std::time::Duration;
use tracing::info;

use crate::profile::ServerProfile;

pub fn launch_server(profile: &ServerProfile) -> anyhow::Result<LmcppServer> {
    log_profile(profile);

    let main_model = std::fs::canonicalize(&profile.model.main)?;

    let build_mode = match profile.toolchain.build_mode.as_str() {
        "build_and_install" => LmcppBuildInstallMode::BuildOrInstall,
        _ => LmcppBuildInstallMode::InstallOnly,
    };
    let toolchain = LmcppToolChain::builder()
        .curl_enabled(profile.toolchain.curl_enabled)
        .build_install_mode(build_mode)
        .build()?;

    let server_args = build_server_args(profile, main_model)?;

    let server = LmcppServerLauncher::builder()
        .toolchain(toolchain)
        .server_args(server_args)
        .load_budget(Duration::from_secs(profile.timeouts.load_secs))
        .download_budget(Duration::from_secs(profile.timeouts.download_secs))
        .load()?;

    info!("서버 준비 완료: {}", profile.api_base_url());
    Ok(server)
}

/// 빌더 typestate 제약 우회: 필수 필드만 빌더로 → build() → Option 필드 직접 대입
fn build_server_args(
    profile: &ServerProfile,
    main_model: std::path::PathBuf,
) -> anyhow::Result<ServerArgs> {
    let mut args = ServerArgs::builder()
        .model(main_model)?
        .host(&profile.server.host)
        .port(profile.server.port)
        .no_webui(profile.server.no_webui)
        .build();

    // [server]
    args.parallel = profile.server.parallel;
    args.timeout = profile.server.timeout;

    // [gpu]
    args.gpu_layers = profile.gpu.layers;
    args.flash_attn = profile.gpu.flash_attn;
    args.device = profile.gpu.device.clone();
    args.main_gpu = profile.gpu.main_gpu;

    // [cpu]
    args.threads = profile.cpu.threads;
    args.threads_batch = profile.cpu.threads_batch;
    args.mlock = profile.cpu.mlock;
    args.no_mmap = profile.cpu.no_mmap;

    // [context]
    args.ctx_size = profile.context.ctx_size;
    args.batch_size = profile.context.batch_size;
    args.ubatch_size = profile.context.ubatch_size;

    // [cache]
    args.cache_type_k = profile.cache.type_k.clone();
    args.cache_type_v = profile.cache.type_v.clone();
    args.cache_reuse = profile.cache.cache_reuse;
    args.no_kv_offload = profile.cache.no_kv_offload;

    // [speculation]
    if let Some(ref spec) = profile.speculation {
        let draft_model = std::fs::canonicalize(&spec.draft)?;
        args.model_draft = Some(draft_model.to_string_lossy().to_string());
        args.draft_max = spec.draft_max;
        args.draft_min = spec.draft_min;
        args.draft_p_min = spec.draft_p_min;
        args.gpu_layers_draft = spec.gpu_layers_draft;
        args.ctx_size_draft = spec.ctx_size_draft;
    }

    // [inference]
    args.reasoning_format = profile.inference.reasoning_format
        .as_ref().map(|v| parse_reasoning_format(v));
    args.jinja = profile.inference.jinja;
    args.chat_template = profile.inference.chat_template.clone();
    args.special_tokens = profile.inference.special_tokens;

    // [debug]
    args.verbose_prompt = profile.debug.verbose_prompt;
    args.verbose = profile.debug.verbose;
    args.log_file = profile.debug.log_file.clone();

    Ok(args)
}

fn parse_reasoning_format(s: &str) -> ReasoningFormat {
    match s.to_lowercase().as_str() {
        "deepseek" => ReasoningFormat::Deepseek,
        "auto" => ReasoningFormat::Auto,
        _ => ReasoningFormat::None,
    }
}

fn log_profile(profile: &ServerProfile) {
    info!("서버 시작: {}", profile.profile.name);
    info!("  모델: {}", profile.model.main);
    if let Some(ngl) = profile.gpu.layers {
        info!("  GPU layers: {}{}", ngl,
            if profile.gpu.flash_attn { " +flash_attn" } else { "" });
    }
    let ctx = profile.context.ctx_size.map_or("default".into(), |v| v.to_string());
    let batch = profile.context.batch_size.map_or("default".into(), |v| v.to_string());
    info!("  ctx: {}, batch: {}", ctx, batch);
    let ck = profile.cache.type_k.as_deref().unwrap_or("default");
    let cv = profile.cache.type_v.as_deref().unwrap_or("default");
    info!("  cache: K={}, V={}", ck, cv);
    if let Some(ref spec) = profile.speculation {
        info!("  드래프트: {} (max {})", spec.draft, spec.draft_max.unwrap_or(16));
    }
    if profile.debug.verbose_prompt { info!("  verbose_prompt: 활성"); }
}
