# CLAUDE.md — local-inference-bench

## Project Overview

Rust CLI tool for benchmarking local LLMs (via llama-server) across a **3-axis framework**:

- **Server Profile** — llama-server configuration (model, GPU, context, speculation, toolchain)
- **RIG Feature** — LLM API pattern (completion, agent, tool calling, multi-turn, RAG)
- **Prompt Set** — Test scenarios (NPC dialogue, tool tests, RAG queries)

Each experiment runs one (profile × feature × prompt_set) combination, collecting latency, token throughput, and success metrics into timestamped result directories.

Documentation and comments are in **Korean**. Commit messages use Korean.

## Tech Stack

- **Language**: Rust (edition 2024, nightly-2026-03-20)
- **LLM Client**: rig-core 0.31 with `derive` feature (agents, tool calling, multi-turn)
- **Server**: llama-server (managed via `lmcpp-toolchain-cli` structure)
- **Async**: tokio 1.x
- **Embeddings**: bge-m3-onnx-rust (git dependency)
- **CLI**: clap 4 (derive)
- **Errors**: anyhow + thiserror 2.0
- **Logging**: tracing + tracing-subscriber (env-filter)
- **Serialization**: serde + serde_json + toml
- **Time/ID**: chrono (timestamps) + uuid v4 (experiment IDs)

## Repository Structure

```
src/
├── main.rs                  # CLI entry (serve / run / list subcommands)
├── profile.rs               # ServerProfile — TOML loading, validation (12 sections)
├── feature.rs               # Feature — dispatch by type (completion/agent/agent_tool/multi_turn/rag)
├── server.rs                # launch_server() — builds CLI args, resolves llama-server binary
├── chat.rs                  # Unified chat wrappers: chat(), chat_with_tools(), chat_with_rag() → ChatResult
├── rag.rs                   # VectorStore (Dense+Sparse), SearchStrategy (DenseOnly/Hybrid/ColBERT), knowledge loader
├── experiment/
│   ├── mod.rs
│   ├── runner.rs            # ExperimentRunner — orchestrates server + feature execution
│   ├── prompt_set.rs        # PromptSet — loads prompt TOML with repeat count
│   └── store.rs             # ExperimentStore — result persistence (experiment.json + runs.jsonl)
└── tools/
    ├── mod.rs
    ├── kung_fu.rs           # lookup_kung_fu — martial arts knowledge lookup
    ├── relationship.rs      # check_relationship — NPC affinity lookup
    └── inventory.rs         # check_inventory — NPC item lookup

config/
├── profiles/*.toml          # Server configs (gemma-4b, gemma3-4b, gemma3-12b, gemma4-e4b, etc.)
├── features/*.toml          # Feature definitions (basic-completion, agent-tool, rag-hybrid, etc.)
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
  → launch_server (resolve binary → spawn llama-server)
  → openai::CompletionsClient (via .completions_api())
  → match feature_type:
      completion/agent → chat()
      agent_tool       → chat_with_tools() (uses CheckInventory, etc.)
      multi_turn       → chat() with shared history
      rag              → chat_with_rag() (BGE-M3 encode → VectorStore search → preamble merge)
  → ExperimentStore (experiment.json metadata + runs.jsonl per-prompt logs)
```

## Critical Design Decisions

1. **`.completions_api()` is mandatory** — llama-server only supports `/v1/chat/completions`. Always use `client.completions_api()` when building the RIG client.

2. **Preamble over context for RAG** — Use `.preamble(system + search_results)` instead of `.context()`. The `.context()` method wraps content in `<file>` XML tags that confuse local models.

3. **`jinja = true` for tool calling** — llama-server needs the `--jinja` flag to parse tool schemas in chat templates. Set in profile TOML under `[inference]`.

4. **Token estimation** — `tokens_generated` in RunRecord uses CJK character count + English word count × 1.3. 실제 API usage (`output_tokens`, `input_tokens`)는 ChatResult에서 받아 `extra` 필드에 저장.

5. **History by reference** — `chat()`, `chat_with_tools()`, `chat_with_rag()` 모두 `&mut Vec<Message>` 참조. RIG가 user/assistant 메시지를 자동 push.

6. **Memory-Efficient RAG** — `RagDocument`는 dense, sparse만 영구 저장. ColBERT 리랭킹 시에는 후보 문서를 실시간으로 `embedder.encode()`하여 메모리 절약 (계산 비용과 트레이드오프).

## Configuration System

### Profiles (`config/profiles/*.toml`)
12 sections: `[profile]`, `[server]`, `[model]`, `[gpu]`, `[cpu]`, `[context]`, `[cache]`, `[speculation]`, `[inference]`, `[debug]`, `[timeouts]`, `[toolchain]`.

### Features (`config/features/*.toml`)
Dispatch key: `feature.type` → `completion`, `agent`, `agent_tool`, `multi_turn`, `rag`.

### Prompt Sets (`config/prompts/*.toml`)
Contains `system_prompt`, `repeat` count, and `[[prompts]]` (id, label, user).

## Adding New Components

**New profile**: Create `config/profiles/<name>.toml`. Sections are optional (defaults used) except `[profile]` and `[model]`.

**New feature**: Create `config/features/<name>.toml`. Ensure `type` matches one of the 5 variants.

**New tool**: Implement `rig::tool::Tool` in `src/tools/`, re-export in `mod.rs`, and register in `ExperimentRunner::run_agent_tool`.

## Code Conventions

- **Error handling**: `anyhow::Result<T>` for app logic, `thiserror` for custom errors.
- **Logging**: `tracing::info!()` for major events. Control via `RUST_LOG`.
- **Async**: All core execution is async via tokio.
- **Naming**: Rust snake_case; TOML keys use snake_case.

## Build Notes

- **Stable Rust supported**: Rust edition 2024 is now stable.
- **CRT mismatch workaround**: `.cargo/config.toml` sets `CXXFLAGS = "/MD"` for ONNX Runtime compatibility.
- **Dependency**: `bge-m3-onnx-rust` is a git dependency.
- **llama-server**: Binary is resolved via `%APPDATA%/llama_cpp_toolchain` or `binary_path` in profile.


## Testing

Manual via `cargo run -- run`. Results inspected with `tools/view-runs.py` or `tools/jsonl-viewer.html`.
