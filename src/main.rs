mod chat;
mod experiment;
mod feature;
mod profile;
mod rag;
mod server;
mod tools;

use clap::{Parser, Subcommand};
use tracing::info;

#[derive(Parser)]
#[command(name = "local-inference-bench")]
#[command(about = "로컬 LLM 실험 벤치: Server Profile × RIG Feature × Prompt Set")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// 서버 프로파일로 llama-server만 띄우기
    Serve {
        /// 프로파일 이름 (config/profiles/*.toml)
        profile: String,
    },
    /// 실험 실행: 서버 띄우고 → 기능 × 프롬프트 세트 테스트
    Run {
        /// 서버 프로파일 이름
        profile: String,
        /// RIG 기능 이름 (config/features/*.toml)
        feature: String,
        /// 프롬프트 세트 이름 (config/prompts/*.toml)
        prompt_set: String,
    },
    /// 등록된 프로파일, 기능, 프롬프트 세트 목록
    List {
        /// profiles | features | prompts | experiments
        #[arg(default_value = "all")]
        what: String,
        /// 실험 필터: --profile qwen3-8b
        #[arg(long)]
        profile: Option<String>,
        /// 실험 필터: --feature agent-preamble
        #[arg(long)]
        feature: Option<String>,
        /// 실험 필터: --prompt-set npc-dialogue
        #[arg(long)]
        prompt_set: Option<String>,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { profile: name } => {
            let prof = profile::ServerProfile::load_by_name(&name)?;
            info!("=== 서버 시작: {} ===", prof.profile.name);
            let _handle = server::launch_server(&prof)?;
            info!("종료: Ctrl+C");
            tokio::signal::ctrl_c().await?;
        }

        Commands::Run { profile: prof_name, feature: feat_name, prompt_set: ps_name } => {
            let prof = profile::ServerProfile::load_by_name(&prof_name)?;
            let feat = feature::Feature::load_by_name(&feat_name)?;
            let ps = experiment::PromptSet::load_by_name(&ps_name)?;

            info!("=== 실험 벤치 ===");
            info!("  서버:     {} ({})", prof.profile.name, prof.profile.description);
            info!("  기능:     {} ({})", feat.feature.name, feat.feature.description);
            info!("  프롬프트: {} ({})", ps.prompt_set.name, ps.prompt_set.description);

            // 서버 시작
            let handle = server::launch_server(&prof)?;

            // 실험 실행
            let runner = experiment::ExperimentRunner::new(prof, feat, ps)
                .with_server_log(handle.log_file);
            let store = runner.run().await?;

            info!("결과 저장: {}", store.dir.display());
        }

        Commands::List { what, profile: pf, feature: ff, prompt_set: ps } => {
            match what.as_str() {
                "profiles" | "all" => {
                    let profiles = profile::ServerProfile::list_all()?;
                    println!("\n[서버 프로파일] ({}개)", profiles.len());
                    for (name, p) in &profiles {
                        let spec = if p.is_speculative() { " +spec" } else { "" };
                        println!("  {} - {}{}", name, p.profile.description, spec);
                    }
                    if what == "profiles" { return Ok(()); }
                }
                _ => {}
            }
            match what.as_str() {
                "features" | "all" => {
                    let features = feature::Feature::list_all()?;
                    println!("\n[RIG 기능] ({}개)", features.len());
                    for (name, f) in &features {
                        println!("  {} ({}) - {}", name, f.feature_type(), f.feature.description);
                    }
                    if what == "features" { return Ok(()); }
                }
                _ => {}
            }
            match what.as_str() {
                "prompts" | "all" => {
                    let sets = experiment::PromptSet::list_all()?;
                    println!("\n[프롬프트 세트] ({}개)", sets.len());
                    for (name, s) in &sets {
                        println!("  {} - {} ({}개 프롬프트, {}회 반복)",
                            name, s.prompt_set.description, s.prompts.len(), s.repeat);
                    }
                    if what == "prompts" { return Ok(()); }
                }
                _ => {}
            }
            match what.as_str() {
                "experiments" | "all" => {
                    let exps = experiment::ExperimentStore::list_experiments(
                        pf.as_deref(), ff.as_deref(), ps.as_deref()
                    )?;
                    println!("\n[실험 결과] ({}개)", exps.len());
                    for e in &exps {
                        let status = match &e.summary {
                            Some(s) => format!("{}/{} 성공, 평균 {:.0}ms, {:.1}tok/s",
                                s.successful_runs, s.total_runs, s.avg_latency_ms, s.avg_tokens_per_sec),
                            None => "진행중".to_string(),
                        };
                        println!("  {} | {} × {} × {} | {}",
                            e.created_at.format("%m/%d %H:%M"),
                            e.profile_name, e.feature_name, e.prompt_set_name, status);
                    }
                }
                _ => {}
            }
        }
    }

    Ok(())
}
