# local-inference-bench

로컬 LLM 실험 벤치. **Server Profile × RIG Feature × Prompt Set** 3축 조합으로 체계적 실험을 수행하고 결과를 관리합니다.

## 사용법

```bash
# 등록된 프로파일/기능/프롬프트 목록 확인
cargo run -- list

# 서버만 띄우기
cargo run -- serve gemma-4b

# 실험 실행 (3축 지정)
cargo run -- run gemma-4b agent-tool npc-tool-test

# RAG 실험 예시
cargo run -- run gemma3-4b rag-hybrid npc-rag-test

# 실험 결과 목록 확인 및 필터링
cargo run -- list experiments --profile gemma3-4b
cargo run -- list experiments --feature rag-colbert
```

## 3축 구조

```text
Server Profile       ×  RIG Feature         ×  Prompt Set
(어떤 서버 설정)         (어떤 RIG 기능)          (어떤 프롬프트)
─────────────────    ───────────────────    ─────────────────
gemma3-4b            basic-completion       npc-dialogue
gemma3-12b-spec      agent-preamble         npc-tool-test
qwen3-8b             agent-tool             npc-rag-test
gemma4-e4b           multi_turn             (추가 가능)
(추가 가능)           rag (Dense/Hybrid/ColBERT)
```

## 디렉토리 구조

```text
config/
├── profiles/    서버 설정 조합 (모델, 배치, 캐시, 스펙큘레이션, 툴체인)
├── features/    RIG 기능 정의 (completion, agent, tool, multi-turn, rag)
├── prompts/     프롬프트 세트 (NPC 대사, 도구 테스트, RAG 쿼리 등)
└── data/        RAG 지식 데이터 (npc-knowledge.toml)

data/experiments/    실험 결과 (gitignored)
└── {timestamp}_{profile}_{feature}_{promptset}/
    ├── experiment.json      3축 스냅샷 + 요약 통계 (성공률, 지연시간, TPS)
    ├── runs.jsonl           개별 프롬프트/응답 상세 로그 (usage, history 포함)
    └── server_filtered.log  llama-server 로그 (DEBUG 제외 필터링)
```

## 핵심 설계

1. **RIG OpenAI API 호환**: `llama-server`의 `/v1/chat/completions` 엔드포인트를 사용하여 도구 호출과 멀티턴 대화를 수행합니다.
2. **BGE-M3 RAG**: 
   - `DenseOnly`: 의미 기반 검색
   - `Hybrid`: Dense + Sparse(키워드) 결합 검색
   - `ColBERTRerank`: ColBERT 기반 정밀 리랭킹
3. **토큰 추정 및 Usage 기록**: ChatResult의 실제 사용량 기록과 함께, 한글/영어 특성을 고려한 자체 토큰 추정 알고리즘을 사용합니다.

## 개발 및 빌드

- **Rust 에디션**: 2024 (Stable 채널 지원)
- **주요 라이브러리**:
  - `rig-core`: LLM 클라이언트 프레임워크
  - `bge-m3-onnx-rust`: 인프로세스 ONNX 임베딩 (RAG용)
  - `tokio`: 비동기 런타임
- **서버 환경**: `%APPDATA%/llama_cpp_toolchain`에 설치된 `llama-server`를 자동으로 탐색하거나 프로파일에 지정된 경로를 사용합니다.
