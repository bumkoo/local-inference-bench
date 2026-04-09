# 프로파일 작성 가이드

## 변경이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-03-21 | 초안 작성. TOML 구조, 시나리오별 예시(빠른 테스트/품질 우선/스펙큘레이션), 벤치마크 비교 전략 |
| 2026-03-21 | 미연결 필드 섹션 제거 — batch_size, cache_type_k/v 모두 server.rs 연결 완료 |

---

프로파일은 `config/profiles/*.toml`에 위치하며, 하나의 llama-server 설정 조합을 정의합니다.

---

## 기본 구조

```toml
[profile]
name = "my-profile"          # 고유 이름 (CLI에서 사용)
description = "설명"          # 목록에서 표시될 설명

[server]
host = "127.0.0.1"
port = 8081
ctx_size = 4096              # 컨텍스트 크기
batch_size = 512             # 프롬프트 처리 배치 크기
parallel = 1                 # 동시 처리 슬롯 수
no_webui = true              # Web UI 비활성화

[model]
main = "..\\models\\모델파일.gguf"

[cache]
type_k = "f16"               # KV 캐시 K 타입: f16, q8_0, q4_0
type_v = "f16"               # KV 캐시 V 타입

# 선택: speculative decoding (없으면 비활성)
[speculation]
draft = "..\\models\\드래프트모델.gguf"
draft_max = 24

[timeouts]
load_secs = 120              # 모델 로딩 대기 시간
download_secs = 0            # 0 = 다운로드 비활성화

[toolchain]
build_mode = "install_only"  # install_only | build_or_install
curl_enabled = false
```

---

## 현재 미연결 필드 — 없음

모든 TOML 필드가 `server.rs`를 통해 `ServerArgs`에 정상 전달됩니다.

---

## 프로파일 예시: 시나리오별

### 빠른 테스트 (작은 모델, 짧은 컨텍스트)

```toml
[profile]
name = "gemma-1b-quick"
description = "Gemma 1B 빠른 반복 테스트용"

[server]
host = "127.0.0.1"
port = 8081
ctx_size = 2048
batch_size = 256
parallel = 1
no_webui = true

[model]
main = "..\\models\\gemma-3-1b-it.Q4_K_M.gguf"

[cache]
type_k = "q8_0"
type_v = "q8_0"

[timeouts]
load_secs = 30
download_secs = 0

[toolchain]
build_mode = "install_only"
curl_enabled = false
```

### 품질 우선 (큰 모델, 넉넉한 컨텍스트)

```toml
[profile]
name = "qwen3-8b-quality"
description = "Qwen3 8B 품질 테스트, f16 캐시, 긴 컨텍스트"

[server]
host = "127.0.0.1"
port = 8081
ctx_size = 8192
batch_size = 1024
parallel = 1
no_webui = true

[model]
main = "..\\models\\Qwen3-8B-Q4_K_M.gguf"

[cache]
type_k = "f16"
type_v = "f16"

[timeouts]
load_secs = 120
download_secs = 0

[toolchain]
build_mode = "install_only"
curl_enabled = false
```

### Speculative Decoding (속도 최적화)

```toml
[profile]
name = "gemma-12b-spec"
description = "Gemma 12B + 1B 드래프트 speculative decoding"

[server]
host = "127.0.0.1"
port = 8081
ctx_size = 4096
batch_size = 512
parallel = 1
no_webui = true

[model]
main = "..\\models\\gemma-3-12b-it-Q3_K_M.gguf"

[cache]
type_k = "q8_0"
type_v = "q8_0"

[speculation]
draft = "..\\models\\gemma-3-1b-it.Q4_K_M.gguf"
draft_max = 24

[timeouts]
load_secs = 180
download_secs = 0

[toolchain]
build_mode = "install_only"
curl_enabled = false
```

---

## 벤치마크 비교 전략

같은 프롬프트 세트로 다음 축을 비교:

1. **모델 크기**: 1B → 4B → 8B → 12B (같은 패밀리)
2. **양자화**: Q3_K_M → Q4_K_M → Q5_K_M (같은 모델)
3. **KV 캐시**: f16 → q8_0 → q4_0 (같은 설정)
4. **Speculation**: 없음 → draft_max 8 → 16 → 24 → 32
5. **컨텍스트**: 2048 → 4096 → 8192 (VRAM 한계 확인)
6. **Thinking 모드**: thinking ON → thinking OFF (Gemma 4 등 `chat_template_kwargs` 지원 모델)

```bash
# 예: Gemma 4 thinking 모드 비교
cargo run -- run gemma4-e4b-think   basic-completion npc-dialogue
cargo run -- run gemma4-e4b-nothink basic-completion npc-dialogue

# 예: 캐시 타입 비교
cargo run -- run qwen3-8b-f16cache   basic-completion npc-dialogue
cargo run -- run qwen3-8b-q8cache    basic-completion npc-dialogue
cargo run -- run qwen3-8b-q4cache    basic-completion npc-dialogue

# 결과 비교
cargo run -- list experiments --profile qwen3-8b
```
