# CLAUDE.md — local-inference-bench

## Project Overview

Rust CLI tool for benchmarking local LLMs (via llama.cpp) across a **3-axis framework**:

- **Server Profile** — llama-server configuration (model, GPU, context, speculation)
- **RIG Feature** — LLM API pattern (completion, agent, tool calling, multi-turn, RAG)
- **Prompt Set** — Test scenarios (NPC dialogue, tool tests, RAG queries)

Each experiment runs one (profile × feature × prompt_set) combination, collecting latency, token throughput, and success metrics into timestamped result directories.

Documentation and comments are in **Korean**. Commit messages use Korean.

## Tech Stack

- **Language**: Rust (edition 2024, nightly-2026-03-20)
- **LLM Client**: rig-core 0.31 with `derive` feature (agents, tool calling, multi-turn)
- **Server**: lmcpp 0.1 (llama.cpp server management)
- **Async**: tokio
- **Embeddings**: bge-m3-onnx-rust (local path dependency `../bge-m3-onnx-rust`)
- **CLI**: clap 4 (derive)
- **Errors**: anyhow + thiserror 2.0
- **Logging**: tracing + tracing-subscriber (env-filter)
- **Serialization**: serde + serde_json + toml
- **Time/ID**: chrono (timestamps) + uuid v4 (experiment IDs)

## Repository Structure

```
src/
├── main.rs                  # CLI entry (serve / run / list subcommands)
├── profile.rs               # ServerProfile — TOML loading, validation
├── feature.rs               # Feature — dispatch by type (completion/agent/agent_tool/multi_turn/rag)
├── server.rs                # launch_server() — builds ServerArgs from profile, wraps lmcpp
├── chat.rs                  # Unified chat wrappers: chat(), chat_with_tools(), chat_with_rag() → ChatResult
├── rag.rs                   # VectorStore (Dense+Sparse만 저장), SearchStrategy (DenseOnly/Hybrid/ColBERTRerank), knowledge loader
├── experiment/
│   ├── mod.rs
│   ├── runner.rs            # ExperimentRunner — orchestrates server + feature execution per prompt
│   ├── prompt_set.rs        # PromptSet — loads prompt TOML with repeat count
│   └── store.rs             # ExperimentStore — result persistence (experiment.json + runs.jsonl)
└── tools/
    ├── mod.rs
    ├── kung_fu.rs           # lookup_kung_fu — martial arts knowledge lookup
    ├── relationship.rs      # check_relationship — NPC affinity lookup
    └── inventory.rs         # check_inventory — NPC item lookup

config/
├── profiles/*.toml          # Server configs (gemma-4b, gemma-12b, gemma-12b-spec, qwen3-8b)
├── features/*.toml          # Feature definitions (8 files: basic-completion, agent-preamble, agent-tool, multi-turn, rag, rag-dense, rag-hybrid, rag-colbert)
├── prompts/*.toml           # Prompt sets (npc-dialogue, npc-tool-test, npc-rag-test)
└── data/npc-knowledge.toml  # RAG knowledge base (11 NPC documents)

docs/                        # Technical documentation (Korean)
├── build-issues.md          # CRT mismatch / ONNX Runtime issues
├── embedding-search.md      # BGE-M3 vector search strategies
├── logging.md               # Tracing/logging configuration
├── models.md                # GGUF model download links
├── profile-guide.md         # Server profile creation guide
├── rig-notes.md             # RIG framework API notes
├── server-options.md        # llama-server CLI options
└── test-coverage-analysis.md # Testing coverage analysis

examples/
└── score_dist.rs            # Score distribution example

tools/
├── view-runs.py             # Python console viewer for experiment results
└── jsonl-viewer.html        # Web-based JSONL viewer

data/experiments/            # Output directory (gitignored) — timestamped experiment results
```

## Key Commands

```bash
# Run an experiment (3-axis combination)
cargo run -- run <profile> <feature> <prompt-set>
# Example: cargo run -- run gemma-4b agent-tool npc-dialogue

# Start server only (no experiment)
cargo run -- serve <profile>

# List registered components or results
cargo run -- list                                    # all categories
cargo run -- list profiles
cargo run -- list experiments --profile gemma-4b     # filtered

# Code quality
cargo fmt
cargo clippy

# View experiment results
python3 tools/view-runs.py [path] [limit]
```

## Architecture & Data Flow

```
TOML configs → CLI (clap) → ExperimentRunner
  → launch_server (lmcpp → llama-server)
  → openai::CompletionsClient (via Client::builder().build()?.completions_api())
  → match feature_type:
      completion/agent → chat()  (run_agent는 run_completion에 위임)
      agent_tool       → chat_with_tools()
      multi_turn       → chat() with shared history
      rag              → chat_with_rag() (BGE-M3 embed → VectorStore search → preamble merge)
  → ExperimentStore (experiment.json metadata + runs.jsonl per-prompt logs)
```

## Critical Design Decisions

1. **`.completions_api()` is mandatory** — llama-server only supports `/v1/chat/completions`, not `/v1/responses`. Always use `client.completions_api()` when building the RIG client.

2. **Preamble over context for RAG** — Use `.preamble(system + search_results)` instead of `.context()`. The `.context()` method wraps content in `<file>` XML tags that confuse local models.

3. **`jinja = true` for tool calling** — llama-server needs the `--jinja` flag to parse tool schemas in chat templates. Set in profile TOML under `[inference]`.

4. **Token estimation** — `tokens_generated` in RunRecord uses CJK character count + English word count × 1.3 (no tokenizer dependency). 실제 API usage (`output_tokens`, `input_tokens` 등)는 ChatResult에서 받아 `extra` JSON 필드에 별도 저장.

5. **History by reference** — `chat()`, `chat_with_tools()`, `chat_with_rag()` 모두 `&mut Vec<Message>` 참조. RIG가 user/assistant 메시지를 자동 push. No cloning.

6. **ColBERT 벡터 미저장** — `RagDocument`는 dense, sparse만 저장. ColBERT 리랭킹 시 후보 문서를 `embedder.encode()`로 재인코딩하여 메모리 절약 (계산 비용 트레이드오프).

## Configuration System

### Profiles (`config/profiles/*.toml`)
12 sections: `[profile]`, `[server]`, `[model]`, `[gpu]`, `[cpu]`, `[context]`, `[cache]`, `[speculation]`, `[inference]`, `[debug]`, `[timeouts]`, `[toolchain]`.

### Features (`config/features/*.toml`)
Dispatch key: `feature.type` → one of: `completion`, `agent`, `agent_tool`, `multi_turn`, `rag`. Only the matching params section is populated.

### Prompt Sets (`config/prompts/*.toml`)
Contains `system_prompt`, `max_tokens`, `repeat` count, and a `[[prompts]]` array with `id`, `label`, `context`, `user` fields.

## Adding New Components

**New profile**: Create `config/profiles/<name>.toml` with required sections. Refer to `docs/profile-guide.md`.

**New feature**: Create `config/features/<name>.toml` with `[feature]` meta (including `type`) and matching params section. If adding a new feature type, add the variant in `feature.rs` and handler in `experiment/runner.rs`.

**New prompt set**: Create `config/prompts/<name>.toml` with `[prompt_set]` meta, `system_prompt`, `repeat`, and `[[prompts]]` entries.

**New tool**: Add a Rust file in `src/tools/`, implement the `rig::tool::Tool` trait, re-export in `src/tools/mod.rs`, and register it in the agent builder in `experiment/runner.rs`.

## Code Conventions

- **Error handling**: `anyhow::Result<T>` for application errors, `thiserror` for custom error types in tools
- **Logging**: `tracing::info!()` with structured messages; control via `RUST_LOG` env var (e.g., `RUST_LOG="local_inference_bench=info,rig::completions=trace"`)
- **Config serialization**: TOML for input configs, JSON for experiment metadata, JSONL for per-run logs
- **Async**: All experiment execution is async via tokio
- **Naming**: Rust snake_case; TOML keys use snake_case with hyphens in filenames

## Build Notes

- **Nightly Rust required**: `nightly-2026-03-20` (see `rust-toolchain.toml`)
- **CRT mismatch workaround**: `.cargo/config.toml` sets `CXXFLAGS = "/MD"` for ONNX Runtime compatibility
- **Local path dependency**: `bge-m3-onnx-rust` is at `../bge-m3-onnx-rust` (sibling directory)
- **Model files**: GGUF models go in `models/` (gitignored). See `docs/models.md` for download links
- **Known build issues**: Documented in `docs/build-issues.md`

## Testing

No automated test suite or CI/CD pipeline. Testing is manual via `cargo run -- run` experiments. Results are inspected with `tools/view-runs.py` or `tools/jsonl-viewer.html`.

## Experiment Output

Each run produces a directory at `data/experiments/{timestamp}_{profile}_{feature}_{promptset}/`:
- `experiment.json` — Full config snapshots + summary stats (success rate, avg latency, avg tokens/sec)
- `runs.jsonl` — One JSON object per prompt execution (latency, tokens, full prompt/response, extra metadata)
- `server_filtered.log` — llama-server logs with DEBUG lines removed
