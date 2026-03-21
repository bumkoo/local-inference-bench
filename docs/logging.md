# LLM 통신 로깅 방법

## 변경이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-03-21 | 초안 작성. RIG/llama-server 양쪽 로깅 방법 정리, 3가지 구현 방안 비교 |

---

## 로깅 레벨

LLM과의 통신은 두 단계로 구성됩니다:

```
[RIG Agent]                          [llama-server]
    │                                     │
    ├─ messages JSON 구성 ──────────────► │
    │  (OpenAI 호환 포맷)                  ├─ chat template 적용
    │                                     ├─ 최종 프롬프트 생성
    │                                     ├─ 토큰화 + 추론
    │  ◄────────────────── 응답 JSON ────┤
    │                                     │
```

| 레벨 | 무엇을 볼 수 있나 | 활성화 방법 |
|---|---|---|
| RIG 요청 | messages 배열, model, temperature 등 | `RUST_LOG=rig::completions=trace` |
| RIG 응답 | choices, usage(토큰 수), finish_reason | `RUST_LOG=rig::completions=trace` |
| llama-server 최종 프롬프트 | chat template 적용 후 실제 텍스트 | `verbose_prompt = true` |
| llama-server 상세 | 토큰화, 샘플링 과정 | `verbose = true` (매우 상세) |

---

## 방안 1: 환경변수로 즉시 활성화 (가장 간단)

### 사용법

```bash
# RIG 요청/응답 로깅 (messages JSON 전체)
RUST_LOG="local_inference_bench=info,rig::completions=trace" cargo run -- run qwen3-8b basic-completion npc-dialogue

# Windows PowerShell
$env:RUST_LOG="local_inference_bench=info,rig::completions=trace"; cargo run -- run qwen3-8b basic-completion npc-dialogue
```

### 장점
- 코드 변경 없음
- RIG가 이미 내장한 tracing 로그 활용

### 단점
- 콘솔에만 출력 (파일 저장 안 됨)
- llama-server 쪽 chat template은 별도로 확인 필요
- 실험 결과(runs.jsonl)에 포함되지 않음

---

## 방안 2: 프로파일에 verbose_prompt 추가 + 로그 파일 저장

### 프로파일 TOML 확장

```toml
[server]
# ... 기존 설정 ...
verbose_prompt = true    # chat template 적용 후 프롬프트 출력
log_file = "data/experiments/server.log"  # 서버 로그 파일 저장

[debug]
rig_trace = true         # RIG 요청/응답 TRACE 레벨 활성화
```

### server.rs에서 verbose_prompt 전달

```rust
// ServerArgs 빌더에 추가
.verbose_prompt(profile.server.verbose_prompt)
```

### 장점
- 프로파일별로 로깅 on/off 제어 가능
- llama-server의 chat template 출력을 파일로 저장
- 실험과 함께 로그 보관

### 단점
- llama-server 로그와 실험 결과가 별도 파일
- llama-server 로그는 텍스트라 파싱 필요

---

## 방안 3: runs.jsonl에 요청 JSON 직접 저장 (가장 체계적)

### RunRecord 확장

```rust
pub struct RunRecord {
    // ... 기존 필드 ...
    
    /// RIG가 보낸 OpenAI 포맷 요청 전체
    pub request_body: Option<serde_json::Value>,
    
    /// llama-server가 반환한 응답 전체
    pub response_body: Option<serde_json::Value>,
}
```

### 구현 방법

RIG의 `CompletionRequest`를 직접 직렬화하는 것은 어렵지만 (RIG 내부 타입),
reqwest 미들웨어 또는 커스텀 HttpClient를 사용하여 HTTP 레벨에서 캡처 가능:

```rust
// 접근 1: reqwest-middleware로 요청/응답 캡처
// rig-core는 reqwest-middleware를 선택적 의존성으로 지원

// 접근 2: 간단한 HTTP 프록시 레이어
// llama-server 앞에 로깅 프록시를 두고 통과시킴

// 접근 3: RIG의 tracing subscriber를 커스텀하여 TRACE 로그를 파일로 저장
// tracing-subscriber의 Layer를 사용하여 rig::completions 타겟만 파일로 라우팅
```

### 장점
- 실험 결과와 요청/응답이 한 파일에 통합
- 나중에 프롬프트 분석, 비교에 직접 활용 가능
- jsonl-viewer에서 바로 확인 가능

### 단점
- runs.jsonl 파일 크기 증가
- 구현 복잡도 가장 높음

---

## 권장: 방안 1 + 2 조합 → 이후 방안 3

### 1단계 (즉시): 환경변수 + verbose_prompt

```bash
# 개발/디버깅 시
$env:RUST_LOG="local_inference_bench=info,rig::completions=trace"
cargo run -- run qwen3-8b basic-completion npc-dialogue
```

프로파일에 `verbose_prompt` 옵션 추가:
```toml
[server]
verbose_prompt = true
```

### 2단계 (다음): tracing-subscriber 파일 라우팅

`rig::completions` 타겟의 TRACE 로그를 실험 디렉토리에 `rig-trace.log`로 별도 저장.
기존 콘솔 출력(INFO)에는 영향 없음.

### 3단계 (향후): RunRecord에 요청 JSON 포함

reqwest 미들웨어 또는 커스텀 HttpClient로 요청/응답 캡처하여 runs.jsonl에 통합.

---

## llama-server chat template 확인 팁

### 방법 1: verbose_prompt로 확인

서버 콘솔에 다음과 같이 출력됨:
```
prompt: '<|im_start|>system\n당신은 무협 RPG의 NPC입니다.\n...<|im_end|>\n<|im_start|>user\n대협, 처음 뵙겠소...<|im_end|>\n<|im_start|>assistant\n'
```

### 방법 2: /v1/chat/completions 수동 호출

```bash
curl http://127.0.0.1:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [
      {"role": "system", "content": "테스트 시스템 프롬프트"},
      {"role": "user", "content": "안녕"}
    ],
    "max_tokens": 10
  }'
```

### 방법 3: /props 엔드포인트

```bash
curl http://127.0.0.1:8081/props
```
모델의 chat template, 기본 설정 등을 JSON으로 반환.

### 방법 4: jinja 활성화 후 chat_template 확인

프로파일에 `jinja = true` 설정 시 Jinja2 기반 템플릿 사용.
모델 파일에 내장된 chat template이 그대로 적용됨.
