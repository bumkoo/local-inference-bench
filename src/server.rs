use lmcpp::{LmcppBuildInstallMode, LmcppServer, LmcppServerLauncher, LmcppToolChain, ServerArgs};
use std::time::Duration;
use tracing::info;

use crate::profile::ServerProfile;

/// 프로파일 기반으로 llama-server를 시작
pub fn launch_server(profile: &ServerProfile) -> anyhow::Result<LmcppServer> {
    info!("서버 시작: {}", profile.profile.name);
    info!("  모델: {}", profile.model.main);
    if profile.is_speculative() {
        let spec = profile.speculation.as_ref().unwrap();
        info!("  드래프트: {} (max {})", spec.draft, spec.draft_max);
    }
    info!("  ctx: {}, batch: {}, parallel: {}",
        profile.server.ctx_size, profile.server.batch_size, profile.server.parallel);
    info!("  cache: K={}, V={}", profile.cache.type_k, profile.cache.type_v);

    let main_model = std::fs::canonicalize(&profile.model.main)?;

    // 툴체인
    let build_mode = match profile.toolchain.build_mode.as_str() {
        "build_and_install" => LmcppBuildInstallMode::BuildOrInstall,
        _ => LmcppBuildInstallMode::InstallOnly,
    };

    let toolchain = LmcppToolChain::builder()
        .curl_enabled(profile.toolchain.curl_enabled)
        .build_install_mode(build_mode)
        .build()?;

    // 서버 인자 (speculation 유무에 따라 분기)
    let server_args = if let Some(ref spec) = profile.speculation {
        let draft_model = std::fs::canonicalize(&spec.draft)?;
        ServerArgs::builder()
            .model(main_model)?
            .host(&profile.server.host)
            .port(profile.server.port)
            .no_webui(profile.server.no_webui)
            .ctx_size(profile.server.ctx_size as u64)
            .batch_size(profile.server.batch_size as u64)
            .parallel(profile.server.parallel as u64)
            .cache_type_k(&profile.cache.type_k)
            .cache_type_v(&profile.cache.type_v)
            .model_draft(draft_model.to_string_lossy().to_string())
            .draft_max(spec.draft_max as u64)
            .build()
    } else {
        ServerArgs::builder()
            .model(main_model)?
            .host(&profile.server.host)
            .port(profile.server.port)
            .no_webui(profile.server.no_webui)
            .ctx_size(profile.server.ctx_size as u64)
            .batch_size(profile.server.batch_size as u64)
            .parallel(profile.server.parallel as u64)
            .cache_type_k(&profile.cache.type_k)
            .cache_type_v(&profile.cache.type_v)
            .build()
    };

    // 런처 시작
    let server = LmcppServerLauncher::builder()
        .toolchain(toolchain)
        .server_args(server_args)
        .load_budget(Duration::from_secs(profile.timeouts.load_secs))
        .download_budget(Duration::from_secs(profile.timeouts.download_secs))
        .load()?;

    info!("서버 준비 완료: {}", profile.api_base_url());
    Ok(server)
}
