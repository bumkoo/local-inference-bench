# 임베딩 저장 및 유사도 검색 설계

## 개요

local-inference-bench의 RAG 검색은 BGE-M3 ONNX 모델을 인프로세스로 실행하여 3종 벡터(Dense, Sparse, ColBERT)를 생성하고, 전략 패턴으로 검색 방식을 교체할 수 있는 구조로 설계되었다. 벡터 DB 없이 인메모리 브루트포스로 동작하며, 향후 LanceDB 등으로 전환할 때 전략 구조를 그대로 유지할 수 있도록 Union 기반 파이프라인을 채택하였다.

## BGE-M3 모델

BGE-M3(BAAI General Embedding Multi-lingual Multi-granularity Multi-functionality)은 하나의 인코딩으로 3종 벡터를 동시에 생성하는 임베딩 모델이다. 본 프로젝트에서는 양자화된 ONNX 모델(`model_quantized.onnx`)을 `ort` 크레이트로 CPU에서 실행한다.

| 벡터 종류 | 차원 | 특성 | 저장 여부 |
|-----------|------|------|-----------|
| Dense | 1024 | 정규화된 실수 벡터. 의미적 유사도 측정의 주축 | O |
| Sparse | {token_id: weight} | BM25 대체. 키워드 매칭에 강함 | O |
| ColBERT | [seq_len × 1024] | 토큰별 벡터. 정밀한 토큰 수준 매칭 | X (리랭킹 시 재인코딩) |


### ColBERT 벡터를 저장하지 않는 이유

ColBERT는 토큰별 1024차원 벡터를 생성하므로 저장 비용이 가장 크다. 100토큰 문서 기준으로 Dense는 4KB, Sparse는 수백 bytes인 반면, ColBERT는 약 400KB이다. 문서 1,000건이면 ColBERT 저장만으로 약 400MB, 10,000건이면 4GB에 달한다.

ColBERT는 전략 3(ColBERT Rerank)에서만 사용되며, 그때도 Union 합집합으로 축소된 후보 문서(일반적으로 5~15개)에 대해서만 필요하다. 따라서 적재 시점에 저장하지 않고, 리랭킹 시점에 후보 문서의 원본 텍스트를 재인코딩하여 ColBERT 벡터를 즉시 생성한 뒤 사용 후 버린다. 같은 텍스트를 같은 모델로 인코딩하면 동일한 벡터가 나오므로 검색 결과에 차이는 없다.

재인코딩 비용은 문서당 약 100ms(CPU, i7-9700K 기준)이며, 후보 10개면 약 1초이다. LLM 추론 시간(7~12초)에 비하면 관리 가능한 수준이다.

## 유사도 함수

### Dense: 코사인 유사도

```
cosine_similarity(a, b) = dot(a, b) / (||a|| × ||b|| + 1e-9)
```

BGE-M3의 Dense 벡터는 정규화되어 있으므로 코사인 유사도 결과가 0~1 범위에 분포한다. 분모의 1e-9은 제로 벡터 방지용이다. 동일 도메인(무협) 문서 간에는 공통 어휘로 인해 바닥 유사도가 0.27~0.35 수준이며, 관련 문서는 0.45~0.65 범위이다. 일반 RAG(뉴스, 위키 등)에서는 무관 문서가 0.1~0.2로 떨어지는 것과 다르다.

### Sparse: 토큰 가중치 곱 합산 (Dot Product)

```
sparse_dot_product(a, b) = Σ (a[token_id] × b[token_id])  (공통 토큰에 대해서만)
```

Sparse 벡터는 {token_id: weight} 형태의 희소 맵이다. 질문과 문서 사이에 공통 토큰이 없으면 점수가 0.0이 된다. 점수 범위가 정규화되어 있지 않으므로(0~0.25 수준), 문서 길이와 토큰 다양성에 따라 절대값이 달라진다. 정답 문서가 Sparse 0인 경우도 존재하므로(예: "검"이 토크나이저에 의해 다른 토큰으로 분절), Sparse 점수에 독립 threshold를 거는 것은 적합하지 않다.

### ColBERT: MaxSim (Multi-Vector)

```
max_sim(Q, D) = (1/|Q|) × Σ_q max_d cosine_similarity(q, d)
```

쿼리의 각 토큰 벡터(q)에 대해, 문서의 모든 토큰 벡터(d) 중 가장 높은 코사인 유사도를 찾고, 이를 쿼리 토큰 수로 나눈 평균이다. 토큰 수준의 정밀한 매칭이 가능하지만, "옥교룡"처럼 고유명사 토큰이 강하게 매칭되면 문서 내용의 실제 관련성과 무관하게 점수가 올라갈 수 있다.

## 문서 적재

```
텍스트 → BgeM3Embedder::encode() → BgeM3Output { dense, sparse, colbert }
                                          ↓
                                    RagDocument { id, text, dense, sparse }
                                    (colbert는 저장하지 않음)
```

`VectorStore::add_batch()`는 각 문서에 대해 `encode()`를 호출하여 Dense+Sparse만 저장한다. 11개 NPC 지식 문서 기준으로 전체 적재에 약 1.1초가 소요된다.

## 검색 전략

3개 전략은 TOML 설정의 `[rag.strategy]` 섹션에서 `type` 필드로 선택한다. `SearchStrategy` 열거형이 serde 태그 기반 역직렬화를 사용하므로, 새 전략 추가 시 열거형 변형만 추가하면 된다.

### 전략 1: Dense Only

```
Dense 전체 스캔 → top-K 선별 → threshold(기본 0.38) 필터 → 결과 반환
```

가장 빠르고 안정적인 전략이다. Dense 코사인 유사도만 사용하며, top-K를 먼저 선별한 뒤 threshold를 후적용한다. threshold를 먼저 걸면 관련 문서가 K개 미만으로 줄어들 수 있고, top-K를 먼저 잡으면 최소 K개 후보를 확보한 상태에서 노이즈만 제거할 수 있다.

실험 결과 top-1 정확도가 6개 프롬프트 중 5/6으로 3개 전략 중 가장 높았다. Sparse의 공통 토큰 과반응 문제가 없어서 도메인 공통 어휘(검, 강호, 무공 등)가 많은 NPC 지식 데이터에 적합하다.

**TOML 설정 예시:**

```toml
[rag.strategy]
type = "dense_only"
top_k = 3
threshold = 0.38
```

### 전략 2: Hybrid (Union → 가중 합산 리랭킹)

```
1차-A: Dense 전체 스캔 → top-K → threshold(기본 0.38) 필터   ← 주축, 노이즈 제거
1차-B: Sparse 전체 스캔 → top-K/2                           ← 보조, threshold 없음
       → 합집합 (중복 제거, 출처 추적)
2차:   합집합 내 Sparse min-max 정규화
       → Dense×0.7 + Sparse_norm×0.3 합산 → 정렬 → top-K 반환
```

Dense와 Sparse를 독립적으로 쿼리한 뒤 합집합을 구성하고, 합집합에 대해서만 가중 합산으로 리랭킹한다.

Dense에는 threshold를 적용하여 0.38 미만인 무관 문서를 걸러내고, Sparse에는 threshold를 걸지 않는다. Sparse에 threshold를 적용하지 않는 이유는, Sparse dot product의 범위가 정규화되어 있지 않고, 정답 문서가 Sparse 0인 경우가 있기 때문이다(예: "검"이라는 단어가 토크나이저에 의해 질문과 문서에서 다른 토큰으로 분절). Sparse는 키워드 보조 역할이므로 top-K/2(candidate_k의 절반)만 선별한다.

합집합 내에서 Sparse 점수를 min-max 정규화하는 이유는, Dense 코사인 유사도는 0~1 범위이지만 Sparse dot product는 범위가 다르기 때문이다. 합집합 내 최대/최소로 0~1 범위로 맞춘 뒤 가중 합산한다.

**TOML 설정 예시:**

```toml
[rag.strategy]
type = "hybrid"
candidate_k = 10        # Dense에서 최대 10개, Sparse에서 최대 5개
top_k = 3               # 최종 반환 수
dense_threshold = 0.38
dense_weight = 0.7
sparse_weight = 0.3
```

### 전략 3: ColBERT Rerank (Union → 재인코딩 → MaxSim 리랭킹)

```
1차:   (전략 2와 동일한 union_candidates)
       Dense top-K(threshold) ∪ Sparse top-K/2 → 합집합
2차:   합집합 후보 문서만 encode() 재호출 → ColBERT 벡터 즉시 생성
       → MaxSim 리랭킹 → top-K 반환
```

1차 단계는 전략 2와 동일한 `union_candidates()`를 공유한다. 차이는 2차 리랭킹에서 가중 합산 대신 ColBERT MaxSim을 사용하는 것이다. 합집합 후보 문서의 원본 텍스트를 `encode()`로 재인코딩하여 ColBERT 벡터를 생성하고, 쿼리의 ColBERT 벡터와 MaxSim 점수를 계산하여 순위를 매긴다.

재인코딩 비용은 합집합 크기에 비례한다. threshold 필터링으로 합집합이 5~10개 수준으로 축소된 경우 약 0.5~1.0초이다. LLM 추론 시간(7~12초) 대비 10% 이내의 추가 비용이다.

**TOML 설정 예시:**

```toml
[rag.strategy]
type = "colbert_rerank"
candidate_k = 10        # Dense에서 최대 10개, Sparse에서 최대 5개
top_k = 3
dense_threshold = 0.38
```

## Union 설계 원칙

### Hybrid 합산 점수는 인덱싱할 수 없다

Dense×0.7 + Sparse×0.3이라는 합산 점수는 두 점수 함수의 선형 결합이므로, 쿼리가 바뀔 때마다 두 점수가 동시에 바뀌어 사전에 합산 점수 기준으로 정렬해 둘 수 없다. 따라서 전수 스캔 방식의 Hybrid는 문서 수가 커질수록 병목이 된다.

실제 대규모 벡터 DB(Milvus, Qdrant, Weaviate 등)에서 "hybrid search"라고 부르는 것은 Union 방식이다. Dense ANN 인덱스(HNSW)에서 top-K를 뽑고, Sparse 역인덱스에서 top-K를 뽑고, 두 결과의 합집합에 대해서만 합산 점수를 계산한다. 각 인덱스가 독립적으로 O(log N) 수준으로 동작하므로 스케일링이 가능하다.

### 벡터 DB 전환 시 매핑

현재 `union_candidates()` 내부의 브루트포스 스캔을 인덱스 쿼리로 교체하면 나머지 전략 구조가 그대로 유지된다.

```
현재 (인메모리 브루트포스):
  Dense:  Vec<RagDocument> 순회 → cosine_similarity() → 정렬 → top-K
  Sparse: Vec<RagDocument> 순회 → sparse_dot_product() → 정렬 → top-K/2

향후 (벡터 DB):
  Dense:  HNSW ANN 인덱스 쿼리 → top-K
  Sparse: 역인덱스(inverted index) 쿼리 → top-K/2
  → 이후 합집합 → 리랭킹은 동일
```

## Threshold 설계

### Dense Threshold (기본 0.38)

Dense 코사인 유사도에만 적용한다. NPC 지식 데이터(무협 도메인) 11개 문서에 대한 점수 분포 분석 결과, 관련 문서는 0.45~0.65 범위, 무관 문서는 0.27~0.38 범위에 분포한다. 0.38은 이 두 범위의 경계에 해당하며, 적대관계 질문에서 11개 중 8개를 걸러내고 배경 질문에서도 8개를 걸러내는 등 효과적인 노이즈 제거를 보여주었다.

0.45(엄격)로 올리면 배경 질문에서 top-1(0.456)만 간신히 통과하고 top-2 이하가 잘린다. 0.30(느슨)으로 내리면 도메인 공통 어휘 때문에 대부분의 문서가 통과하여 필터링 효과가 거의 없다.

범용 RAG 시스템에서는 보통 0.5~0.6을 사용하지만, 동일 도메인 문서만 모아놓은 경우 바닥 유사도가 높으므로 그에 맞춰 낮춰야 한다. 문서가 늘어나면서 도메인 폭이 넓어지면 재조정이 필요하다.

### Sparse에 Threshold를 걸지 않는 이유

Sparse dot product는 0 아니면 값이 있는 이진적 분포를 보인다. 질문과 문서 사이에 공통 토큰이 없으면 0.0이고, 있으면 0.01~0.25 범위이다. 66개 쌍(6 질문 × 11 문서) 중 30%가 0.0이며, 정답 문서가 0.0인 경우도 있다(예: 소지품 질문에서 inv_li_mubai의 Sparse 점수가 0.0). 점수의 절대값이 문서 길이, 토큰 다양성, 토크나이저 분절 방식에 따라 크게 달라지므로, 특정 threshold 값이 범용적으로 의미를 갖기 어렵다.

따라서 Sparse는 top-K/2 선별만 적용하고, threshold는 걸지 않는다.

### 적용 위치

threshold가 걸리는 위치는 전략마다 다르다.

전략 1(Dense Only)에서는 top-K 선별 후 최종 결과에 threshold를 후적용한다.

전략 2, 3(Hybrid, ColBERT)에서는 1차 Union 단계의 Dense 쪽에만 threshold를 적용한다. Dense top-K를 먼저 선별하고 threshold 미달 문서를 제거한 뒤, Sparse top-K/2(threshold 없음)와 합집합을 구성한다. 2차 리랭킹 후에는 별도 threshold를 적용하지 않는다. 리랭킹 점수(가중 합산 또는 MaxSim)는 Dense 코사인 유사도와 범위가 다르므로 동일 threshold를 적용할 수 없고, 리랭킹 점수용 별도 threshold를 도입하면 설정 복잡도만 올라간다.

## 실험 결과 (Gemma 4B, NPC 지식 11개 문서)

### 전략별 top-1 정확도

| 프롬프트 | 이상적 top-1 | Dense | Hybrid(7:3) | ColBERT |
|----------|-------------|-------|-------------|---------|
| 무공 질문 | kungfu_wudang_sword | ✅ 0.596 | ✅ 0.717 | ✅ 0.512 |
| 관계 질문 | rel_yu_jiaolong | ❌ bg_yu.. | ❌ bg_yu.. | ❌ bg_yu.. |
| 적대 관계 | rel_jade_fox | ✅ 0.601 | ✅ 0.721 | ✅ 0.582 |
| 소지품 | inv_li_mubai | ✅ 0.530 | ❌ kungfu_fly.. | ✅ 0.458 |
| 복합 질문 | rel_yu_jiaolong | ✅ 0.645 | ✅ 0.752 | ❌ bg_yu.. |
| 배경 질문 | bg_li_mubai | ✅ 0.456 | ❌ rel_yu_xiu.. | ✅ 0.422 |
| **정확도** | | **5/6** | **3/6** | **4/6** |

Dense가 가장 안정적이며, Hybrid는 Sparse의 도메인 공통 토큰 과반응으로 소지품/배경 질문에서 오답이 발생한다. ColBERT는 토큰 수준 MaxSim이 고유명사("옥교룡") 토큰에 과도하게 반응하여 복합 질문에서 Dense보다 낮은 정확도를 보였다.

관계 질문은 3개 전략 모두 오답이다. "옥교룡이라는 아이를 아시오?"에서 `bg_yu_jiaolong`(배경 문서)과 `rel_yu_jiaolong`(관계 문서) 모두 "옥교룡"을 포함하여 의미적으로 구분이 어려운 케이스로, 전략 문제라기보다 문서 설계의 한계이다.

### threshold 필터링에 의한 합집합 크기 변화

| 프롬프트 | Dense(threshold≥0.38) | Sparse(top-5) | 합집합 |
|----------|----------------------|---------------|--------|
| 무공 질문 | 6개 | 5개 | 6개 |
| 관계 질문 | 7개 | 5개 | 7개 |
| 적대 관계 | 3개 | 5개 | 5개 |
| 소지품 | 9개 | 5개 | 10개 |
| 복합 질문 | 10개 | 5개 | 10개 |
| 배경 질문 | 3개 | 5개 | 5개 |

적대관계와 배경 질문에서 Dense threshold가 11개 중 8개를 걸러내어 합집합이 5개로 축소되었다. ColBERT 전략에서는 재인코딩 대상이 절반으로 줄어 비용이 직접 절감된다.

### Dense 점수 분포 (11개 NPC 지식 문서)

| 프롬프트 | top-1 | top-3 평균 | 나머지 평균 | 갭 |
|----------|-------|-----------|-----------|-----|
| 무공 질문 | 0.596 | 0.514 | 0.372 | 0.141 |
| 관계 질문 | 0.597 | 0.546 | 0.374 | 0.171 |
| 적대 관계 | 0.601 | 0.501 | 0.343 | 0.158 |
| 소지품 | 0.530 | 0.514 | 0.416 | 0.098 |
| 복합 질문 | 0.645 | 0.615 | 0.451 | 0.164 |
| 배경 질문 | 0.456 | 0.434 | 0.337 | 0.097 |

소지품과 배경 질문은 top-3과 나머지 사이의 갭이 0.1 미만으로, 점수만으로는 관련/무관 문서 구분이 어렵다. 이는 모든 문서가 동일 무협 도메인이라 Dense 유사도 분포가 압축되어 있기 때문이다.

## 관련 파일

| 파일 | 역할 |
|------|------|
| `src/rag.rs` | SearchStrategy 열거형, VectorStore, union_candidates(), 3개 전략 구현 |
| `src/chat.rs` | chat_with_rag() — 검색 결과를 preamble에 합쳐서 LLM 호출 |
| `src/feature.rs` | RagParams 구조체 — TOML에서 전략 설정 역직렬화 |
| `src/experiment/runner.rs` | run_rag() — 실험 실행 및 결과 기록 |
| `config/features/rag-dense.toml` | 전략 1 설정 |
| `config/features/rag-hybrid.toml` | 전략 2 설정 |
| `config/features/rag-colbert.toml` | 전략 3 설정 |
| `config/data/npc-knowledge.toml` | NPC 지식 데이터 (11개 문서) |
| `config/prompts/npc-rag-test.toml` | RAG 테스트 프롬프트 세트 (6개) |
| `examples/score_dist.rs` | 전체 문서 점수 분포 확인용 스크립트 |
| `bge-m3-onnx-rust/src/lib.rs` | BGE-M3 ONNX 임베더, cosine_similarity, sparse_dot_product, max_sim |


## 설계 변경 이력

이 섹션은 개발 과정에서 발생한 설계 변경과 그 근거를 시간 순으로 기록한다. 각 변경은 실험 결과와 구조적 분석에 기반한다.

### v1: Dense Only 기반 (초기 구현)

RagDocument에 Dense 벡터(1024차원)만 저장하고, `encode_dense()`로 임베딩하여 코사인 유사도로 전수 스캔 검색하는 단순한 구조로 시작하였다. VectorStore의 `search()`는 `query_dense`와 모든 문서의 `dense` 벡터를 비교하여 top-K를 반환하였다. 이 시점에서 `chat_with_rag()`는 `top_k: usize`를 직접 파라미터로 받았다.

Gemma 4B 기준 6/6 성공, 64~75 tok/s로 기본 RAG 파이프라인이 정상 동작함을 확인하였다. UTF-8 바이트 경계 panic 버그(`&r.text[..80]`)를 `chars().take(80)` 방식으로 수정하였다.

### v2: 3종 벡터 저장 + 전략 패턴 도입

`SearchStrategy` 열거형을 태그 기반 serde 역직렬화로 설계하고, DenseOnly / DenseSparseHybrid / ColBERTRerank 3개 전략을 구현하였다. RagDocument에 Dense/Sparse/ColBERT 3종 벡터를 모두 저장하도록 확장하였다. `feature.rs`의 `RagParams`에서 `top_k: usize`를 `strategy: SearchStrategy`로 교체하고, `chat_with_rag()`도 `&SearchStrategy`를 받도록 변경하였다.

이 시점의 Hybrid는 **전체 문서에 대해** Dense 코사인 유사도와 Sparse dot product를 전부 계산하고 `Dense×0.6 + Sparse×0.4`로 합산하여 정렬하는 전수 스캔 방식이었다. ColBERT Rerank의 1차 단계도 이 Hybrid 전수 스캔을 `candidate_k`개로 자르는 방식이었다.

실험 결과, Hybrid가 Dense보다 높은 점수를 보였으나(0.758 vs 0.596), 소지품 질문과 배경 질문에서 Sparse의 도메인 공통 토큰("검", "강호") 과반응으로 무관한 문서가 top-1에 올라오는 문제가 발생하였다. Dense 5/6, Hybrid 3/6, ColBERT 4/6으로 Dense가 가장 안정적이었다.

### v3: Union 방식 전환

**문제 인식**: Hybrid 합산 점수(`Dense×w1 + Sparse×w2`)는 두 점수 함수의 선형 결합이므로 사전 인덱싱이 불가능하다. 문서 100,000개 기준으로 1차 단계에서 10만 건 전수 스캔이 필요하며, 이 구조는 벡터 DB로 전환할 때 그대로 옮길 수 없다.

또한 ColBERT Rerank의 1차 단계가 Hybrid 전수 스캔이므로, 문서 11개에 `candidate_k=10`이면 Hybrid와 ColBERT의 후보군이 거의 동일하여 ColBERT만의 차별화 가치가 없었다.

**변경**: Dense top-K와 Sparse top-K를 독립적으로 뽑아 합집합(Union)을 구성하는 방식으로 전환하였다. `union_candidates()` 함수가 각 인덱스를 독립 쿼리하고, 합집합에 대해서만 리랭킹을 수행한다. Hybrid와 ColBERT가 동일한 `union_candidates()`를 1차 단계로 공유하되, 리랭킹 함수(가중 합산 vs MaxSim)만 다르게 적용된다.

이 구조에서 벡터 DB 전환 시, `union_candidates()` 내부의 브루트포스를 HNSW ANN 쿼리 + 역인덱스 쿼리로 교체하면 전체 전략 구조가 그대로 유지된다.

동시에 기본 가중치를 Dense 0.6 / Sparse 0.4에서 **Dense 0.7 / Sparse 0.3**으로 변경하여 Sparse 과반응을 억제하였다. 전략 이름도 `dense_sparse_hybrid`에서 `hybrid`로 간결하게 변경하였다.

### v4: ColBERT 벡터 미저장, 리랭킹 시 재인코딩

**문제 인식**: ColBERT 벡터는 토큰별 1024차원 배열로, 문서당 저장 비용이 Dense(4KB)의 100배 수준(100토큰 기준 약 400KB)이다. 문서 수천 건이면 ColBERT 저장만으로 기가바이트 단위가 된다. 그런데 ColBERT는 전략 3에서만 사용되며, 그때도 Union 합집합으로 축소된 후보 문서(5~15개)에 대해서만 필요하다.

**변경**: RagDocument에서 `colbert: Vec<Vec<f32>>` 필드를 제거하고, 적재 시 Dense+Sparse만 저장하도록 변경하였다. ColBERT Rerank 리랭킹 시 후보 문서의 원본 텍스트를 `encode()`로 재인코딩하여 ColBERT 벡터를 즉시 생성한 뒤 사용 후 버린다. `VectorStore::search()`와 `query()`에 `embedder: &mut BgeM3Embedder` 파라미터를 추가하여 ColBERT 전략에서 재인코딩할 수 있도록 하였다.

재인코딩 비용은 후보 11개 기준 약 1.1초(CPU i7-9700K)이며, LLM 추론 시간(7~12초) 대비 관리 가능한 수준이다. 검색 결과는 동일하다(같은 텍스트를 같은 모델로 인코딩하면 같은 벡터가 생성됨).

### v5: Dense threshold + Sparse top-K/2 비대칭 Union (현재)

**문제 인식**: 전체 점수 분포 분석(`examples/score_dist.rs`)을 통해, Dense 코사인 유사도는 관련/무관 문서 사이에 0.38 근처 경계가 있으나, Sparse dot product는 정답 문서가 0.0인 경우가 존재하여(예: 소지품 질문의 `inv_li_mubai`) threshold를 걸면 정답을 버리게 되는 것을 확인하였다.

**변경**: Dense와 Sparse의 1차 선별 방식을 비대칭으로 설계하였다.

- Dense: top-K 선별 후 threshold(기본 0.38) 필터. 의미 검색의 주축으로, threshold로 노이즈를 제거한다.
- Sparse: top-K/2 선별, threshold 없음. 키워드 보조 역할로, 정답이 Sparse 0일 수 있으므로 수량만 줄인다.
- 합집합 후 리랭킹(가중 합산 또는 ColBERT MaxSim) → top-K 반환.

전략 설정의 `threshold` 필드를 DenseOnly에서는 `threshold`로, Hybrid/ColBERT에서는 `dense_threshold`로 분리하여 역할을 명확히 하였다. Hybrid/ColBERT의 리랭킹 후에는 별도 threshold를 적용하지 않는다. 리랭킹 점수(가중 합산 또는 MaxSim)는 Dense 코사인 유사도와 범위가 다르므로 동일 threshold를 적용할 수 없다.

실험 결과, 적대관계 질문에서 Dense threshold가 11개 중 8개를 걸러내어 합집합이 5개로 축소되었고, 배경 질문에서도 합집합이 5개로 축소되어 ColBERT 재인코딩 비용이 절반으로 줄었다. 전략별 정확도는 Dense 5/6, Hybrid 3/6, ColBERT 4/6으로 유지되었다.

### 변경 요약

| 버전 | 핵심 변경 | 동기 |
|------|----------|------|
| v1 | Dense Only 기본 RAG | 최소 동작 확인 |
| v2 | 3종 벡터 + 전략 패턴 (전수 스캔 Hybrid) | 검색 품질 실험 |
| v3 | Union 방식 전환 (Dense top-K ∪ Sparse top-K) | Hybrid 인덱싱 불가 문제, 벡터 DB 전환 대비 |
| v4 | ColBERT 벡터 미저장, 리랭킹 시 재인코딩 | 메모리 절감 (문서당 400KB → 0) |
| v5 | Dense threshold + Sparse top-K/2 비대칭 Union | Sparse threshold 부적합, 노이즈 제거 |
