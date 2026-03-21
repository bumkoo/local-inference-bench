# 보유 모델 목록

## 변경이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-03-21 | 초안 작성. 추론 모델 13개, 임베딩 모델 2개, speculative decoding 조합 쌍 정리 |

---

> 경로: `C:\Users\bumko\projects\models\`  
> 프로파일에서는 `..\\models\\파일명.gguf`로 참조

---

## 추론 모델 (텍스트 생성)

| 파일명 | 패밀리 | 크기 | 양자화 | 용도 |
|---|---|---|---|---|
| Qwen3-8B-Q4_K_M.gguf | Qwen3 | 8B | Q4_K_M | 메인 추론 |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf | Llama 3.1 | 8B | Q4_K_M | 메인 추론 |
| gemma-3-12b-it-Q3_K_M.gguf | Gemma 3 | 12B | Q3_K_M | 메인 (큰 모델) |
| gemma-3-12b-it-Q4_K_M.gguf | Gemma 3 | 12B | Q4_K_M | 메인 (큰 모델) |
| google_gemma-3-12b-it-Q3_K_M.gguf | Gemma 3 | 12B | Q3_K_M | 메인 (큰 모델) |
| google_gemma-3-12b-it-IQ4_XS.gguf | Gemma 3 | 12B | IQ4_XS | 극한 양자화 테스트 |
| gemma-3-12b-it-qat-int4-Q4_K_M.gguf | Gemma 3 QAT | 12B | Q4_K_M | QAT 양자화 비교 |
| Gemma-3-12b-it-MAX-HORROR-D_AU-Q3_K_S-imat.gguf | Gemma 3 | 12B | Q3_K_S | imatrix 양자화 |
| gemma-3-4b-it-Q4_K_M.gguf | Gemma 3 | 4B | Q4_K_M | 중간 크기 |
| gemma-3n-E4B-it-Q4_K_M.gguf | Gemma 3n | E4B | Q4_K_M | 경량 모델 |
| gemma-3-1b-it.Q4_K_M.gguf | Gemma 3 | 1B | Q4_K_M | 드래프트 모델 |
| gemma-3-1b-it-q4_0.gguf | Gemma 3 | 1B | Q4_0 | 드래프트 모델 |
| gemma-3-270m-it-Q4_K_M.gguf | Gemma 3 | 270M | Q4_K_M | 초경량 드래프트 |

## 임베딩 모델

| 파일명 | 패밀리 | 크기 | 용도 |
|---|---|---|---|
| embeddinggemma-300M-Q8_0.gguf | EmbeddingGemma | 300M | 벡터 임베딩 |
| embeddinggemma-300m-qat-Q8_0.gguf | EmbeddingGemma QAT | 300M | 벡터 임베딩 |
| bge-m3/ (디렉토리) | BGE-M3 | - | 다국어 임베딩 (ONNX) |

---

## Speculative Decoding 조합 가능 쌍

같은 토크나이저를 사용해야 효과적:

| 메인 모델 | 드래프트 모델 | 비고 |
|---|---|---|
| gemma-3-12b-it-Q3_K_M | gemma-3-1b-it.Q4_K_M | Gemma 패밀리 |
| gemma-3-4b-it-Q4_K_M | gemma-3-1b-it.Q4_K_M | Gemma 패밀리 |
| gemma-3-4b-it-Q4_K_M | gemma-3-270m-it-Q4_K_M | Gemma 패밀리 (초경량 드래프트) |

Qwen3과 Llama 3.1은 현재 드래프트 모델이 없음 → 0.5B~1B 양자화 모델 다운로드 고려
