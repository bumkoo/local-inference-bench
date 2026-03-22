# 빌드 이슈 및 해결 내역

## 환경

- OS: Windows 11
- CPU: Intel Core i7-9700K
- GPU: NVIDIA GeForce RTX 2070 SUPER (8GB)
- Rust: nightly-2026-03-20
- MSVC: Visual Studio Build Tools (x86_64-pc-windows-msvc)

## 1. Nightly 툴체인 요구

**증상**: `llm_models` 크레이트(lmcpp의 의존성)가 `#![feature(f16)]`을 사용하여 stable 채널에서 컴파일 불가.

```
error[E0554]: `#![feature]` may not be used on the stable release channel
 --> llm_models-0.0.3/src/lib.rs:1:1
  |
1 | #![feature(f16)]
  | ^^^^^^^^^^^^^^^^
```

**원인**: `f16` 타입이 Rust nightly에서만 사용 가능한 unstable feature이다. lmcpp → llm_models 의존 체인에서 f16을 사용.

**해결**: `rust-toolchain.toml`을 프로젝트 루트에 생성하여 nightly 고정.

```toml
[toolchain]
channel = "nightly-2026-03-20"
components = ["rustfmt", "clippy"]
```

특정 날짜의 nightly를 고정한 이유는, nightly는 매일 빌드가 달라져서 재현 가능한 빌드를 보장하기 위함이다.

## 2. CRT 불일치 (ort_sys /MD vs esaxx-rs /MT)

**증상**: `bge-m3-onnx-rust` 의존성 추가 후 링크 에러. ort_sys(ONNX Runtime C 바인딩)는 /MD(동적 CRT)로 컴파일되고, esaxx-rs(tokenizers의 의존성, C++ 코드)는 /MT(정적 CRT)로 컴파일되어 Windows MSVC 링커가 CRT 심볼 충돌을 보고.

**원인**: Windows MSVC에서 하나의 바이너리 내 모든 오브젝트 파일은 동일한 CRT 링크 방식(/MD 또는 /MT)을 사용해야 한다. ort_sys는 ONNX Runtime 공유 라이브러리를 로드하므로 /MD가 필수이고, esaxx-rs의 C++ 빌드 스크립트는 기본적으로 /MT를 사용한다.

**해결**: `.cargo/config.toml`에서 `CXXFLAGS` 환경변수를 `/MD`로 설정하여 C++ 코드의 CRT를 동적으로 강제.

```toml
# local-inference-bench/.cargo/config.toml
# CRT 불일치 해결: ort_sys(/MD) vs esaxx_rs(/MT)
# CXXFLAGS만 설정 — esaxx-rs(C++)만 /MD로 강제
# CFLAGS는 건드리지 않음 — aws-lc-sys(CMake) 깨짐 방지
[env]
CXXFLAGS = "/MD"
```

**주의**: `CFLAGS`를 함께 설정하면 `aws-lc-sys`(rustls의 의존성, CMake 빌드)가 깨진다. aws-lc-sys는 CMake에서 자체적으로 CRT 플래그를 관리하며, 외부 CFLAGS가 개입하면 충돌이 발생한다. 따라서 CXXFLAGS만 설정하여 esaxx-rs(C++)만 /MD로 강제하고, C 코드는 각 빌드 스크립트의 기본값을 따르도록 한다.

참고로 `bge-m3-onnx-rust`의 `.cargo/config.toml`은 다른 설정을 사용한다:

```toml
# bge-m3-onnx-rust/.cargo/config.toml
[target.x86_64-pc-windows-msvc]
rustflags = ["-C", "target-feature=-crt-static"]

[env]
CFLAGS = "/MD"
CXXFLAGS = "/MD"
```

bge-m3-onnx-rust 단독 빌드에서는 aws-lc-sys가 의존성에 없으므로 CFLAGS = "/MD"를 설정해도 문제없다. 하지만 local-inference-bench에서 path 의존성으로 통합하면 local-inference-bench의 `.cargo/config.toml`이 적용되므로, 여기서는 CXXFLAGS만 설정한다.

## 3. rig-core 0.31.0 API 변경

**증상**: 여러 API 에러가 동시에 발생.

### 3-1. `from_url` 제거

```
error[E0599]: no function or associated item named `from_url` found
 --> src/experiment/runner.rs:35:38
```

**원인**: rig-core 0.31.0에서 `openai::Client::from_url()`이 제거되고, `ProviderClient` 트레이트 기반으로 변경.

**해결**: `openai::Client::from_env()`로 생성한 뒤 `completions_api(base_url)`로 OpenAI 호환 엔드포인트를 지정.

```rust
let client = openai::Client::from_env();
let completions = client.completions_api(&base_url);
```

### 3-2. `agent` 메서드 trait import 누락

```
error[E0599]: no method named `agent` found for reference `&Client`
help: trait `Prompt` which provides `prompt` is implemented but not in scope
```

**해결**: `use rig::client::CompletionClient;` import 추가.

### 3-3. `PromptResponse` private 모듈 접근

```
error[E0603]: module `prompt_request` is private
 --> src/chat.rs:13:17
  | use rig::agent::prompt_request::PromptResponse;
```

**해결**: `use rig::agent::PromptResponse;`로 변경 (re-export 경로 사용).

### 3-4. `Prompt` 트레이트 import 누락

```
error[E0599]: no method named `prompt` found for struct `Agent`
```

**해결**: `use rig::completion::Prompt;` import 추가.

## 4. lmcpp 0.1.1 타입 불일치

**증상**: `server.rs`에서 lmcpp의 `ServerArgs` 빌더 메서드에 `u32` 값을 전달하면 `u64` 타입 불일치 에러.

```
error[E0308]: mismatched types
 --> src/server.rs:38:19
  | .ctx_size(profile.server.ctx_size)
  |           ^^^^^^^^^^^^^^^^^^^^^^^ expected `u64`, found `u32`
```

**원인**: lmcpp 0.1.1에서 `ctx_size`, `parallel`, `draft_max` 필드가 `Option<u64>`로 정의되어 있으나, 프로젝트의 profile TOML에서 `u32`로 파싱.

**해결**: `.into()` 호출로 u32 → u64 변환.

```rust
.ctx_size(profile.server.ctx_size.into())
.parallel(profile.server.parallel.into())
.draft_max(spec.draft_max.into())
```

## 5. lmcpp API 변경 — BuildAndInstall → BuildOrInstall

**증상**: `LmcppBuildInstallMode::BuildAndInstall` variant가 없음.

```
error[E0599]: no variant named `BuildAndInstall` found
```

**해결**: `LmcppBuildInstallMode::BuildOrInstall`로 변경. lmcpp 0.1.1에서 enum variant 이름이 변경됨.

## 6. lmcpp 반환 타입 변경 — LmcppServer vs LmcppServerLauncher

**증상**: `create_server()` 함수의 반환 타입 불일치.

```
error[E0308]: mismatched types
 --> src/server.rs:60:8
  | Ok(server)
  |    ^^^^^^ expected `LmcppServerLauncher`, found `LmcppServer`
```

**해결**: lmcpp 0.1.1에서 서버 생성 API가 빌더 → 런처 패턴으로 변경됨. `ServerArgs::builder().build()` 후 `.launch()` 호출이 아닌 빌더 결과를 직접 런처로 반환하도록 수정.

## 7. thiserror 크레이트 누락

**증상**: RIG Tool 구현용 에러 타입에서 thiserror 매크로 사용 시 컴파일 에러.

```
error[E0433]: cannot find module or crate `thiserror` in this scope
 --> src/tools/kung_fu.rs:16:17
  | #[derive(Debug, thiserror::Error)]
  |                 ^^^^^^^^^
```

**해결**: `Cargo.toml`에 `thiserror = "2.0.18"` 의존성 추가.

## 8. bge-m3-onnx-rust 라이브러리 크레이트 전환

**증상**: bge-m3-onnx-rust가 바이너리 크레이트(main.rs만 존재)여서 local-inference-bench에서 path 의존성으로 참조할 수 없음.

**해결**: `bge-m3-onnx-rust/src/lib.rs`를 새로 생성하여 라이브러리 인터페이스 공개.

공개 항목:
- `BgeM3Embedder` — ONNX 세션 관리, encode()/encode_dense()
- `BgeM3Output` — dense/sparse/colbert 3종 벡터 출력
- `cosine_similarity()`, `sparse_dot_product()`, `max_sim()` — 유사도 함수

기존 `main.rs`는 테스트용 바이너리로 유지. lib.rs와 main.rs가 공존하여 `cargo run`으로 바이너리 실행, `cargo check --lib`로 라이브러리 검증이 모두 가능.

## 9. UTF-8 바이트 경계 panic

**증상**: RAG 검색 결과의 한글 텍스트를 로그에 출력할 때 런타임 panic.

```
thread 'main' panicked at 'byte index 80 is not a char boundary'
```

**원인**: `&r.text[..80]`으로 문자열을 자를 때, UTF-8 멀티바이트 문자(한글은 3바이트)의 중간 바이트에서 슬라이스를 시도하면 Rust가 panic을 발생시킨다. 한글 텍스트에서 80바이트 지점이 한 글자의 2번째 바이트에 해당하는 경우.

**해결**: 바이트 인덱스 대신 문자 단위 접근으로 변경.

```rust
// 수정 전 — panic 가능
let preview = &r.text[..80];

// 수정 후 — 안전
let preview: String = r.text.chars().take(80).collect();
```

## 10. ort::init() 호출 위치

**증상**: ONNX Runtime 세션 생성 시 ort 초기화가 되어 있지 않으면 경고 또는 에러.

**해결**: `run_rag()` 함수 진입 직후, BgeM3Embedder 생성 이전에 `ort::init().commit()?;`를 호출. ort 2.0에서는 최초 1회 초기화가 필요하며, 중복 호출은 무시된다. 전체 프로세스에서 한 번만 호출하면 되므로, main에서 호출하거나 run_rag() 진입점에서 호출.

## 현재 빌드 설정 요약

### 의존성 체인에서 발생하는 충돌 관계

```
local-inference-bench
├── lmcpp 0.1.1
│   └── llm_models 0.0.3 ← #![feature(f16)] → nightly 필수
├── rig-core 0.31.0
│   └── reqwest → rustls → aws-lc-sys ← CMake 빌드, CFLAGS 개입 시 깨짐
├── bge-m3-onnx-rust (path)
│   ├── ort 2.0.0-rc.12
│   │   └── ort-sys ← /MD (동적 CRT) 필수
│   └── tokenizers 0.21.4
│       └── esaxx-rs ← 기본 /MT (정적 CRT), CXXFLAGS="/MD"로 강제
└── .cargo/config.toml
    └── CXXFLAGS="/MD" (CFLAGS는 설정하지 않음)
```

### 설정 파일

| 파일 | 내용 | 이유 |
|------|------|------|
| `rust-toolchain.toml` | `nightly-2026-03-20` | llm_models의 f16 feature |
| `.cargo/config.toml` | `CXXFLAGS = "/MD"` | ort_sys(/MD) vs esaxx-rs(/MT) CRT 불일치 |
| bge-m3-onnx-rust `.cargo/config.toml` | CFLAGS+CXXFLAGS="/MD", crt-static 비활성 | 단독 빌드용 (통합 시 local-inference-bench 설정이 적용) |

### 이슈 요약

| # | 이슈 | 카테고리 | 해결 |
|---|------|---------|------|
| 1 | llm_models f16 feature | 툴체인 | nightly 고정 |
| 2 | ort_sys vs esaxx-rs CRT | 링크 | CXXFLAGS="/MD" |
| 3 | rig-core 0.31 API 변경 | API | import 경로 수정 |
| 4 | lmcpp u32/u64 타입 | API | .into() 변환 |
| 5 | lmcpp enum 이름 변경 | API | BuildOrInstall |
| 6 | lmcpp 반환 타입 변경 | API | Launcher 패턴 적용 |
| 7 | thiserror 누락 | 의존성 | Cargo.toml 추가 |
| 8 | bge-m3-onnx-rust lib.rs | 구조 | 라이브러리 크레이트 전환 |
| 9 | UTF-8 바이트 경계 panic | 런타임 | chars().take() 사용 |
| 10 | ort 초기화 | 런타임 | ort::init().commit() |
