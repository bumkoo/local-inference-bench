# RIG 활용 노트

> local-inference-bench에서 RIG(rig-core 0.31) 사용 시 주의사항 모음.
> 로컬 LLM(llama-server) 환경 기준.

---

## 1. `.completions_api()` 필수 사용

RIG 0.31부터 OpenAI 클라이언트의 기본 API 경로가 변경됨.

| 메서드 | API 경로 | 비고 |
|---|---|---|
| (기본) | `/v1/responses` | OpenAI Responses API (2025~ 신규) |
| `.completions_api()` | `/v1/chat/completions` | OpenAI Chat Completions API (기존) |

llama-server, Ollama 등 OpenAI 호환 서버는 **`/v1/chat/completions`만 지원**하므로, 반드시 `.completions_api()`를 호출해야 함.

```rust
// ✅ 올바른 사용
let client = openai::Client::builder()
    .api_key("local-no-key")
    .base_url("http://127.0.0.1:8081/v1")
    .build()?
    .completions_api();   // ← 필수

// ❌ 이렇게 하면 /v1/responses로 요청 → llama-server 404
let client = openai::Client::builder()
    .base_url("http://127.0.0.1:8081/v1")
    .build()?;
```

**참고 예제**: `rig-source/rig/rig-core/examples/openai_agent_completions_api.rs`

---

## 2. `.context()` 로컬 LLM 비호환 — `.preamble()`로 통합 권장

RIG의 `.context()`는 내부적으로 다음과 같이 처리됨:

### 동작 방식

1. `.context("텍스트")`를 호출하면 `Document { id: "static_doc_0", text: "텍스트" }` 생성
2. 요청 시 `Document`의 `Display` trait에 의해 `<file>` 태그로 래핑:
   ```
   <file id: static_doc_0>
   텍스트 내용
   </file>
   ```
3. 이 래핑된 문서가 **user 메시지**로 전송됨 (system 메시지가 아님!)

### 실제 전송되는 messages 구조

```json
{
  "messages": [
    {"role": "system", "content": "preamble 내용"},
    {"role": "user", "content": "<file id: static_doc_0>\n캐릭터 설정...\n</file>\n"},
    {"role": "user", "content": "실제 유저 프롬프트"}
  ]
}
```

### 로컬 LLM에서의 문제

- `<file id: ...>` 태그는 RIG 자체 포맷이며, OpenAI GPT 모델 대상으로 설계됨
- Gemma, Qwen 등 로컬 모델은 이 태그를 학습하지 않아 **노이즈로 작용** 가능
- context가 system이 아닌 **user 메시지**로 들어가므로, chat template 적용 시 의도와 다르게 배치될 수 있음

### 권장: `.preamble()`에 통합

```rust
// ✅ 로컬 LLM 권장 방식: preamble에 system_prompt + context 합침
let full_system = format!("{}\n\n{}", system_prompt, context);
let agent = client.agent("local-model")
    .preamble(&full_system)
    .build();

// ❌ 로컬 LLM 비권장: .context() 사용
let agent = client.agent("local-model")
    .preamble(&system_prompt)
    .context(&character_info)   // → <file> 태그 + user 메시지로 전송
    .build();
```

저장 시에는 `runs.jsonl`에 `system_prompt`과 `context`를 별도 필드로 기록하여 증빙 관리.

**소스 참조**: `rig-source/rig/rig-core/src/agent/builder.rs` (context → Document 저장), `rig-source/rig/rig-core/src/completion/request.rs` (Document Display → `<file>` 래핑, normalized_documents → User 메시지 변환)

---

---

## 3. Multi-turn: `.with_history()` 사용 권장 — 수동 컨텍스트 합치기 비권장

RIG의 `.max_turns()`와 `.with_history()`는 다른 기능임.

### `.max_turns(N)` — Tool Calling 왕복 횟수 제한

```rust
let result = agent.prompt("Calculate (3+5)/9")
    .max_turns(20)   // 도구 호출 → 결과 → 재요청 왕복 최대 20번
    .await?;
```

LLM이 텍스트 응답(도구 호출 없음)을 반환하면 루프 즉시 종료.
"사용자 멀티턴 대화"가 아닌 **에이전트 내부 도구 호출 루프** 제어용.

### `.with_history()` — 대화 이력 자동 관리

```rust
let mut history: Vec<Message> = Vec::new();

// 턴 1
let res1 = agent.prompt("대협, 처음 뵙겠소.")
    .with_history(&mut history)
    .await?;
// history = [User("대협, 처음 뵙겠소."), Assistant("이모백이라 하오...")]

// 턴 2 — 이전 대화가 messages 배열에 자동 포함
let res2 = agent.prompt("무공을 알려주시오")
    .with_history(&mut history)
    .await?;
// history = [User, Assistant, User, Assistant] 4개 누적
```

RIG가 내부적으로:
1. `history`에 현재 user 메시지 push
2. system + history 전체를 messages 배열로 전송
3. LLM 응답 후 assistant 메시지도 history에 push

### 수동 방식 vs `.with_history()` 비교

**수동 방식 (이전 구현)**:
- 이전 대화를 system prompt 텍스트에 합침
- LLM이 보는 구조: `[system(이력포함), user]`
- 이력이 role 구분 없이 평문으로 system 안에 섞임
- chat template의 user/assistant 특수 토큰 미적용

**`.with_history()` (현재 구현)**:
- RIG가 messages 배열에 user/assistant 교대로 누적
- LLM이 보는 구조: `[system, user, assistant, user, assistant, ...]`
- chat template이 역할별 특수 토큰을 정확히 적용

### cache_reuse 효과

`.with_history()`가 **더 유리함**.

```
턴1 토큰: [system][user1]
턴2 토큰: [system][user1][assistant1][user2]   ← 턴1 전체가 접두사와 일치!
턴3 토큰: [system][user1][assistant1][user2][assistant2][user3]  ← 턴2 전체가 접두사!
```

이전 턴의 프롬프트 전체가 다음 턴의 접두사이므로, `cache_reuse`가 최대한 효과적으로 동작.

### 증빙 저장 방식

`runs.jsonl`의 `extra.conversation`에 매 턴의 대화 로그를 JSON 배열로 저장:

```json
{
  "turn": 2,
  "total_turns": 5,
  "conversation": [
    {"role": "user", "content": "대협, 처음 뵙겠소."},
    {"role": "assistant", "content": "이모백이라 하오..."},
    {"role": "user", "content": "무공을 알려주시오"},
    {"role": "assistant", "content": "무당검법은..."}
  ]
}
```

`system_prompt` 필드에는 preamble(시스템 프롬프트 + 캐릭터 컨텍스트)이 저장되어, 전체 재현 가능.

### `.chat()` 비사용 — 깊은 복사(deep copy) 문제

RIG Skill(`skills/rig/SKILL.md`)에서는 `.chat()` 패턴을 제안하지만, 사용하지 않음.

```rust
// RIG Skill 제안 패턴 — 비사용
let response = agent.chat("Hi, I'm Alice.", history.clone()).await?;
history.push("Hi, I'm Alice.".into());
history.push(response.into());
```

`.chat()`의 내부 구현 (agent/completion.rs):

```rust
async fn chat(
    &self,
    prompt: impl Into<Message>,
    mut chat_history: Vec<Message>,  // ← &가 아닌 값 전달 = 소유권 이전(move)
) -> Result<String, PromptError> {
    PromptRequest::from_agent(self, prompt)
        .with_history(&mut chat_history)  // 내부에서 with_history 호출
        .await
}
```

**`.chat()`은 `.prompt().with_history()`의 래퍼**이며, 동일 기능을 다른 인터페이스로 제공할 뿐임.

#### 비사용 이유: 소유권 이전 + 깊은 복사

`.chat()`의 시그니처가 `chat_history: Vec<Message>` (참조가 아닌 값)이므로:

1. **소유권 이전**: 호출 시 `history`의 소유권이 `.chat()` 내부로 넘어감
2. **`.clone()` 필수**: 호출 후에도 history를 사용하려면 `history.clone()` 필요
3. **깊은 복사**: `Vec<Message>`의 `.clone()`은 벡터 안의 **모든 Message + 내부 String 텍스트를 전부 복사**
4. **수동 push**: `.chat()` 내부에서 history에 push하지만 복사본에 대한 것이므로 **원본에 미반영** → 호출자가 수동으로 push 필요
5. **턴 누적 비용**: 대화 10턴이면 매 턴마다 10개 메시지 전체를 깊은 복사

#### `.with_history()` 사용 이유

```rust
// ✅ 우리 방식
let mut history: Vec<Message> = Vec::new();
let res = agent.prompt("대협, 처음 뵙겠소.")
    .with_history(&mut history)  // &mut 참조 → 복사 없음
    .await?;
// RIG가 원본 history에 직접 user + assistant push → 수동 관리 불필요
```

`.with_history()`의 시그니처: `history: &'a mut Vec<Message>` (참조)

| | `.chat()` | `.with_history()` |
|---|---|---|
| 전달 방식 | `Vec<Message>` (소유권 이전) | `&mut Vec<Message>` (참조) |
| 복사 | 매 턴마다 `.clone()` 필수 (깊은 복사) | 복사 없음 |
| user/assistant push | 호출자가 **수동** push | RIG가 **자동** push |
| 호출 후 history | 원본에 반영 안 됨 | 원본에 바로 반영 |
| `.max_turns()` 체이닝 | 불가 | 가능 (tool calling 루프 제어) |

#### Chat trait의 설계 의도 — 외부 통합용 인터페이스

RIG GitHub에 `.chat()`의 깊은 복사 관련 이슈/PR은 없음. 이 설계는 의도적.

`Chat` trait은 외부 프레임워크 통합을 위한 인터페이스:
- `TelegramBot<T: Chat>` — Telegram 봇이 에이전트를 감쌀 때 사용 (rig-telegram, Issue #109)
- `TwitterBot<T: Chat>` — Twitter 봇 통합 (Issue #110)
- State Machine 패턴 — `ChatAgentStateMachine` (rig-agent-state-machine-example)

이런 통합에서는:
- history 관리를 봇 프레임워크/상태 머신이 자체적으로 수행
- 매 호출마다 history **스냅샷**을 넘기는 게 오히려 안전 (프레임워크가 동시성 제어)
- 호출 후 원본 history에 자동 반영되면 오히려 부작용 발생 가능

반면 **직접 멀티턴 루프를 작성하는 경우** (우리 runner.rs처럼):
- history를 직접 관리하므로 `.with_history(&mut history)` 패턴이 적합
- clone 비용 없음, 자동 push, `.max_turns()` 체이닝 가능

결론: `.chat()`은 trait 기반 추상화용, `.with_history()`는 직접 제어용. 
**로컬 LLM 벤치마크/게임 NPC 구현에서는 `.with_history()` 사용.**

### 주의사항

- 멀티턴에서 **첫 프롬프트의 context**로 에이전트를 한 번만 생성 (동일 캐릭터 연속 대화)
- 프롬프트마다 다른 캐릭터를 테스트하려면 멀티턴이 아닌 단일턴 반복 사용
- `.with_history()` + `.max_turns()` 조합으로 도구 호출 멀티턴도 가능 (추후 확장)

---

## 4. Tool Calling 시 `jinja = true` 필수

### 문제

RIG에서 `.tool()`로 도구를 등록한 agent가 llama-server에 요청하면, 요청 JSON에 `tools` 파라미터가 포함됨.
llama-server가 `--jinja` 없이 기동된 상태면 500 에러 발생:

```
{"code":500,"message":"tools param requires --jinja flag","type":"server_error"}
```

### 원인

llama-server의 chat template 처리 방식이 두 가지:

| 모드 | 동작 | tool calling |
|---|---|---|
| 기본 (jinja 없음) | C++ 하드코딩된 간이 파서로 처리 | **불가** — 도구 스키마를 프롬프트에 삽입하는 로직 없음 |
| `--jinja` | 모델 GGUF 내장 Jinja2 템플릿을 실제 Jinja 엔진으로 실행 | **가능** — 템플릿 안의 `{% if tools %}...` 로직이 실행됨 |

모델이 tool calling을 지원하더라도 (Gemma 3, Qwen3 등의 chat template에 도구 관련 Jinja 로직이 포함), 
llama-server가 `--jinja`로 기동되지 않으면 그 로직이 실행되지 않음.

즉 **모델이 지원 + 서버가 `--jinja`로 활성화** 둘 다 필요.

### 설정

프로파일 TOML의 `[inference]` 섹션:

```toml
[inference]
jinja = true
```

이 설정이 `server.rs`에서 `args.jinja = true` → lmcpp가 `--jinja` 플래그 전달.

### 참고

- `jinja = true` 활성화 시 Chat format이 `Content-only` → `Generic`으로 변경됨
- BOS 토큰 중복 경고가 나올 수 있음 (기능에 영향 없음)
- 단순 대사 생성 (tool calling 없는 실험)에서는 `jinja = true` 불필요

---

---

## 향후 검토: 도구 호출 모델과 대사 생성 모델 분리

현재 RIG의 agent 루프는 하나의 모델이 도구 호출 판단 + 최종 대사 생성을 모두 수행.
wuxia-rpg에서는 라우터 모델(작은 모델, 빠른 판단)과 대사 모델(큰 모델, 품질 중시)을 분리하는 구조 검토 예정.
RIG 자동 루프 대신 수동 2단계 구현 필요 — 벤치마크가 아닌 게임 프로젝트 단계에서 진행.

---

*이 문서는 RIG 사용 중 발견되는 주의사항을 지속적으로 추가합니다.*
