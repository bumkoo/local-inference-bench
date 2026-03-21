# local-inference-bench

로컬 LLM 실험 벤치. **Server Profile × RIG Feature × Prompt Set** 3축 조합으로 체계적 실험을 수행하고 결과를 관리합니다.

## 사용법

```bash
# 등록된 프로파일/기능/프롬프트 목록 확인
cargo run -- list

# 서버만 띄우기
cargo run -- serve gemma-4b

# 실험 실행 (3축 지정)
cargo run -- run gemma-4b agent-preamble npc-dialogue

# 실험 결과 필터링
cargo run -- list experiments --profile gemma3-4b
cargo run -- list experiments --feature agent-tool
```

## 3축 구조

```
Server Profile       ×  RIG Feature         ×  Prompt Set
(어떤 서버 설정)         (어떤 RIG 기능)          (어떤 프롬프트)
─────────────────    ───────────────────    ─────────────────
qwen3-8b             basic-completion       npc-dialogue
gemma-12b-spec       agent-preamble         (추가 예정)
(추가 가능)           agent-tool
                     multi-turn
```

## 디렉토리 구조

```
config/
├── profiles/    서버 설정 조합 (모델, 배치, 캐시, 스펙큘레이션)
├── features/    RIG 기능 정의 (completion, agent, tool, multi-turn)
└── prompts/     프롬프트 세트 (NPC 대사, 내레이션 등)

data/experiments/    실험 결과 (gitignored)
└── {timestamp}_{profile}_{feature}_{promptset}/
    ├── experiment.json    3축 스냅샷 + 요약 통계
    └── runs.jsonl         개별 프롬프트/응답 로그
```

## 의존성

- `lmcpp` - llama.cpp 서버 관리
- `rig-core` - LLM 클라이언트 프레임워크 (에이전트, 도구, 체인)
