# llama-server 주요 옵션 레퍼런스

## 변경이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-03-21 | 초안 작성. lmcpp 0.1.1 ServerArgs 기준 주요 옵션 약 40개 정리 |
| 2026-03-21 | 미연결 필드(batch_size, cache_type_k/v) server.rs 연결 완료, 확장 우선순위 업데이트 |

---

> lmcpp `ServerArgs`를 통해 설정 가능한 llama-server 옵션 정리  
> 출처: https://github.com/ggml-org/llama.cpp/tree/master/tools/server  
> lmcpp 소스: `lmcpp-0.1.1/src/server/types/start_args.rs`

---

## 1. GPU / 디바이스

### gpu_layers (ngl)
- **타입**: `Option<u64>`
- **설명**: 모델 레이어 중 GPU(VRAM)에 올릴 개수. 나머지는 CPU/RAM에서 처리
- **기본값**: 미설정 시 lmcpp/llama.cpp 기본값 사용
- **영향**: 추론 속도에 가장 큰 영향. GPU 레이어가 많을수록 빠르지만 VRAM 한계 있음
- **RTX 2070 (8GB)**: 
  - 7B Q4 모델 → `99` (전체 GPU 오프로드 가능)
  - 12B Q3 모델 → `20~30` (부분 오프로드, VRAM 여유 확인 필요)
- **실험 포인트**: 같은 모델에서 ngl 값을 바꿔가며 tok/s 비교

### flash_attn (fa)
- **타입**: `bool`
- **설명**: Flash Attention 활성화. 어텐션 연산을 메모리 효율적으로 수행
- **기본값**: `false`
- **영향**: VRAM 사용량 감소 + 긴 컨텍스트에서 속도 향상
- **주의**: 모든 GPU/모델 조합에서 지원되지 않을 수 있음. RTX 2070(Turing)에서는 제한적
- **실험 포인트**: flash_attn on/off에 따른 성능 및 VRAM 사용량 비교
- **벤치마크 (RTX 2070 Super, Gemma 4B Q4_K_M)**:
  - flash_attn=true, cache f16: **~65 t/s** (eval) — flash_attn 단독은 속도 저하 없음
  - flash_attn=true, cache q8_0: **~28 t/s** (eval) — **56% 성능 저하**
  - 원인: q8_0 KV cache 양자화의 역양자화 오버헤드 (flash_attn과 무관)
  - 결론: **RTX 2070에서는 flash_attn=true 가능, cache는 f16(기본) 권장**

### device
- **타입**: `Option<String>`
- **설명**: 사용할 디바이스 지정 (예: `CUDA0`, `CPU`)
- **기본값**: 자동 감지

### main_gpu
- **타입**: `Option<u32>`
- **설명**: 멀티 GPU 환경에서 메인으로 사용할 GPU 인덱스
- **기본값**: `0`
- **참고**: 싱글 GPU 환경에서는 설정 불필요

---

## 2. 컨텍스트 / 배치

### ctx_size (c)
- **타입**: `Option<u64>`
- **설명**: 컨텍스트 크기 (토큰 수). 한 번에 처리할 수 있는 최대 입출력 길이
- **기본값**: `4096`
- **영향**: 클수록 긴 대화 가능하지만 VRAM 사용량 비례 증가
- **실험 포인트**: NPC 대사(짧음) vs 내레이션(길음) 시나리오별 적정값 탐색
- **참고**: KV 캐시 크기 = ctx_size × n_layers × head_dim × 2(K,V) × dtype_size

### batch_size (b)
- **타입**: `Option<u64>`
- **설명**: 프롬프트 처리 시 한 번에 처리하는 토큰 수 (prompt processing batch)
- **기본값**: `2048`
- **영향**: 클수록 프롬프트 처리(prefill) 속도 향상, VRAM 더 사용
- **실험 포인트**: 긴 시스템 프롬프트 사용 시 batch_size 증가 효과

### ubatch_size (ub)
- **타입**: `Option<u64>`
- **설명**: 마이크로 배치 크기. batch_size 내에서 실제 GPU에 보내는 단위
- **기본값**: `512`
- **영향**: batch_size보다 작게 설정하면 VRAM 피크 사용량 감소
- **관계**: `ubatch_size <= batch_size`

### parallel (np)
- **타입**: `Option<u64>`
- **설명**: 동시 처리 가능한 요청 슬롯 수
- **기본값**: `1`
- **영향**: 여러 요청 동시 처리 가능하지만, VRAM을 슬롯 수만큼 분할
- **벤치 용도**: 보통 `1`~`2`. 벤치마크에서는 순차 실행이므로 `1`이 정확한 측정

---

## 3. KV 캐시

### cache_type_k / cache_type_v
- **타입**: `Option<String>`
- **설명**: KV 캐시의 양자화 타입
- **옵션**: `f16`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1`
- **기본값**: `f16`
- **영향**: 
  - `f16` → 가장 정확, VRAM 많이 사용
  - `q8_0` → 품질 거의 동일, VRAM ~50% 절약
  - `q4_0` → 품질 약간 저하, VRAM ~75% 절약
- **실험 포인트**: 같은 모델에서 캐시 양자화에 따른 품질/속도 트레이드오프
- **벤치마크 (RTX 2070 Super, Gemma 4B Q4_K_M)**:
  - f16(기본): **~65 t/s** (eval)
  - q8_0: **~28 t/s** (eval) — 56% 성능 저하, 역양자화 오버헤드가 원인
  - 결론: RTX 2070에서는 **f16(기본) 권장**. VRAM 절약이 필요한 경우에만 양자화 고려.

### cache_reuse
- **타입**: `Option<u64>`
- **설명**: 이전 요청의 KV 캐시를 재사용할 토큰 수
- **기본값**: 미설정
- **영향**: 동일 시스템 프롬프트 반복 시 prefill 시간 절약
- **실험 포인트**: 같은 NPC 페르소나로 연속 대화 시 효과 측정

### no_kv_offload
- **타입**: `bool`
- **설명**: KV 캐시를 GPU에서 CPU로 오프로드하지 않음
- **기본값**: `false`
- **참고**: VRAM이 부족할 때 캐시만 CPU에 두는 옵션 (느려짐)

### defrag_thold
- **타입**: `Option<f32>`
- **설명**: KV 캐시 단편화 임계값. 이 비율 이상 단편화되면 정리
- **기본값**: `-1.0` (비활성)
- **범위**: `0.0` ~ `1.0`

---

## 4. Speculative Decoding (투기적 디코딩)

### model_draft
- **타입**: `Option<String>`
- **설명**: 드래프트 모델 경로 (작고 빠른 모델)
- **원리**: 드래프트 모델이 여러 토큰을 빠르게 추측 → 메인 모델이 한 번에 검증
- **조합 예시**:
  - Qwen2.5-7B(메인) + Qwen2.5-0.5B(드래프트)
  - Gemma-12B(메인) + Gemma-1B(드래프트)
- **주의**: 같은 토크나이저를 사용하는 모델 쌍이어야 효과적

### draft_max
- **타입**: `Option<u64>`
- **설명**: 드래프트 모델이 한 번에 추측할 최대 토큰 수
- **기본값**: `16`
- **영향**: 클수록 공격적으로 추측하지만, reject 시 낭비 증가
- **실험 포인트**: `8`, `16`, `24`, `32`로 변경하며 acceptance rate 비교

### draft_min
- **타입**: `Option<u64>`
- **설명**: 최소 드래프트 토큰 수
- **기본값**: `1`

### draft_p_min
- **타입**: `Option<f32>`
- **설명**: 드래프트 토큰의 최소 수용 확률
- **기본값**: `0.0`
- **영향**: 높이면 품질 향상(reject 증가), 낮추면 속도 향상

### ctx_size_draft
- **타입**: `Option<u64>`
- **설명**: 드래프트 모델의 컨텍스트 크기
- **참고**: 메인 모델과 다르게 설정 가능

### gpu_layers_draft
- **타입**: `Option<u32>`
- **설명**: 드래프트 모델의 GPU 레이어 수
- **실험 포인트**: 드래프트를 전부 GPU에 올리면 투기 속도 최적화

### cache_type_k_draft / cache_type_v_draft
- **타입**: `Option<String>`
- **설명**: 드래프트 모델의 KV 캐시 양자화
- **참고**: 드래프트는 정확도보다 속도가 중요하므로 `q4_0` 고려

---

## 5. CPU / 스레드

### threads (t)
- **타입**: `Option<u64>`
- **설명**: 추론에 사용할 CPU 스레드 수
- **기본값**: 시스템 코어 수
- **참고**: GPU 오프로드 비율이 높으면 큰 영향 없음

### threads_batch (tb)
- **타입**: `Option<u64>`
- **설명**: 배치 처리(프롬프트 인제스트)에 사용할 스레드 수
- **기본값**: threads와 동일
- **실험 포인트**: 배치 스레드를 늘리면 긴 프롬프트 처리 속도 향상 가능

### threads_http
- **타입**: `Option<u64>`
- **설명**: HTTP 서버 스레드 수
- **기본값**: `parallel` 값과 동일
- **참고**: 벤치마크에서는 영향 미미

---

## 6. 메모리 관리

### mlock
- **타입**: `bool`
- **설명**: 모델을 RAM에 고정 (swap 방지)
- **기본값**: `false`
- **영향**: 모델이 swap으로 밀리지 않아 안정적 성능
- **주의**: 시스템 RAM이 충분해야 함

### no_mmap
- **타입**: `bool`
- **설명**: 메모리 맵핑 비활성화. 모델을 직접 RAM에 로드
- **기본값**: `false`
- **영향**: mmap은 lazy loading이라 초기 로드 빠름. no_mmap은 전체 로드 후 시작

---

## 7. 추론 / 생성 설정

### reasoning_format
- **타입**: `Option<ReasoningFormat>` → `none`, `deepseek`, `auto`
- **설명**: 모델의 thinking/reasoning 출력 형식
- **기본값**: 미설정 (모델 기본)
- **영향**: Qwen3, DeepSeek 등 thinking 모델에서 reasoning 토큰 처리 방식 결정
- **실험 포인트**: `none` vs `deepseek`에 따른 NPC 대사 품질 차이

### reasoning_budget
- **타입**: `Option<ReasoningBudget>` → 숫자 또는 특수값
- **설명**: reasoning 토큰 예산 제한

### jinja
- **타입**: `bool`
- **설명**: Jinja2 채팅 템플릿 활성화
- **기본값**: `false`
- **참고**: 모델별 chat template 정확도를 위해 사용

### chat_template
- **타입**: `Option<String>`
- **설명**: 커스텀 채팅 템플릿 지정
- **참고**: 모델 내장 템플릿 대신 사용자 정의 가능

### special_tokens
- **타입**: `bool`
- **설명**: 특수 토큰 처리 활성화
- **기본값**: `false`

---

## 8. RoPE (Rotary Position Embedding)

### rope_freq_base
- **타입**: `Option<f32>`
- **설명**: RoPE 기본 주파수. 컨텍스트 확장에 사용
- **기본값**: 모델 내장값
- **참고**: 모델 학습 시 설정된 값과 다르면 품질 저하 가능

### rope_freq_scale
- **타입**: `Option<f32>`
- **설명**: RoPE 주파수 스케일링 팩터
- **사용처**: 학습 컨텍스트보다 긴 컨텍스트 사용 시

### rope_scaling
- **타입**: `Option<RopeScaling>` → `none`, `linear`, `yarn`
- **설명**: RoPE 스케일링 방식
- **YaRN**: 긴 컨텍스트 확장을 위한 방식. `yarn_orig_ctx`, `yarn_ext_factor` 등과 함께 사용

---

## 9. LoRA / 제어

### lora
- **타입**: `Option<Vec<String>>`
- **설명**: LoRA 어댑터 경로 (여러 개 지정 가능)
- **형식**: `path:scale` (예: `adapter.gguf:0.8`)
- **실험 포인트**: NPC 성격별 fine-tuned LoRA 적용 테스트

### control_vector
- **타입**: `Option<Vec<String>>`
- **설명**: 제어 벡터 파일. 모델 행동을 특정 방향으로 조절
- **형식**: `path:scale`

---

## 10. 멀티모달

### mmproj
- **타입**: `Option<String>`
- **설명**: 멀티모달 프로젝터 파일 경로 (이미지 입력용)
- **참고**: 게임 내 시각 입력 처리에 향후 사용 가능

---

## 11. 기타 서버 설정

### embeddings_only
- **타입**: `bool`
- **설명**: 임베딩 전용 모드 (텍스트 생성 비활성화)
- **참고**: 벡터 DB용 임베딩 모델 서버로 사용 시

### metrics_endpoint
- **타입**: `bool`
- **설명**: Prometheus 메트릭 엔드포인트 활성화 (`/metrics`)
- **참고**: 성능 모니터링에 유용

### slots_endpoint
- **타입**: `bool`
- **설명**: 슬롯 상태 엔드포인트 활성화 (`/slots`)
- **참고**: 서버 상태 모니터링

### timeout
- **타입**: `Option<u64>`
- **설명**: 서버 요청 타임아웃 (초)

---

## 프로파일 TOML 매핑 (현재 → 확장 예정)

현재 `config/profiles/*.toml`에서 사용 중인 필드:

| TOML 섹션.필드 | CLI 플래그 | 상태 |
|---|---|---|
| `server.host` | `--host` | 사용 중 |
| `server.port` | `--port` | 사용 중 |
| `server.parallel` | `--parallel` | 사용 중 |
| `server.no_webui` | `--no-webui` | 사용 중 |
| `server.timeout` | `--timeout` | 사용 중 |
| `model.main` | `--model` | 사용 중 |
| `gpu.layers` | `--gpu-layers` | 사용 중 |
| `gpu.flash_attn` | `--flash-attn` | 사용 중 |
| `gpu.device` | `--device` | 사용 중 |
| `gpu.main_gpu` | `--main-gpu` | 사용 중 |
| `gpu.split_mode` | `--split-mode` | 사용 중 |
| `cpu.threads` | `--threads` | 사용 중 |
| `cpu.threads_batch` | `--threads-batch` | 사용 중 |
| `cpu.threads_http` | `--threads-http` | 사용 중 |
| `cpu.mlock` | `--mlock` | 사용 중 |
| `cpu.no_mmap` | `--no-mmap` | 사용 중 |
| `cpu.numa` | `--numa` | 사용 중 |
| `context.ctx_size` | `--ctx-size` | 사용 중 |
| `context.batch_size` | `--batch-size` | 사용 중 |
| `context.ubatch_size` | `--ubatch-size` | 사용 중 |
| `cache.type_k` | `--cache-type-k` | 사용 중 |
| `cache.type_v` | `--cache-type-v` | 사용 중 |
| `cache.cache_reuse` | `--cache-reuse` | 사용 중 |
| `cache.no_kv_offload` | `--no-kv-offload` | 사용 중 |
| `cache.defrag_thold` | `--defrag-thold` | 사용 중 |
| `speculation.*` | `--model-draft` 등 | 사용 중 |
| `inference.reasoning_format` | `--reasoning-format` | 사용 중 |
| `inference.reasoning_budget` | `--reasoning-budget` | 사용 중 |
| `inference.jinja` | `--jinja` | 사용 중 |
| `inference.chat_template` | `--chat-template` | 사용 중 |
| `inference.special_tokens` | `--special-tokens` | 사용 중 |
| `debug.verbose` | `--verbose` | 사용 중 |
| `debug.verbosity` | `-lv` | 사용 중 |
| `debug.log_file` | `--log-file` | 사용 중 |
| `debug.slots_endpoint` | `--slots` | 사용 중 |
| `debug.metrics_endpoint` | `--metrics` | 사용 중 |

미구현 (향후 확장 후보):
1. `rope_freq_base` / `rope_freq_scale` / `rope_scaling` — RoPE 컨텍스트 확장
2. `lora` — LoRA 어댑터
3. `mmproj` — 멀티모달 프로젝터
4. `control_vector` — 제어 벡터
