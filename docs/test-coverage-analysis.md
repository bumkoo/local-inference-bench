# 테스트 커버리지 분석 및 개선 제안

## 현재 상태

코드베이스에 자동화된 테스트가 **전혀 없음** (0% 커버리지).

- `#[test]` 또는 `#[cfg(test)]` 블록 없음
- `tests/` 통합 테스트 디렉토리 없음
- `Cargo.toml`에 `dev-dependencies` 없음
- 테스트는 수동으로 `cargo run -- run` 실험 실행으로만 수행

총 소스 코드: ~2,221줄 (15개 파일), 모두 테스트 미작성 상태.

---

## 우선순위별 테스트 개선 제안

### 1. [최우선] RAG 검색 알고리즘 (`src/rag.rs`, 437줄)

코드베이스에서 가장 복잡한 모듈로, 3가지 검색 전략의 정확성이 검증되지 않은 상태.

**테스트 대상:**

| 함수 | 테스트 내용 | 난이도 |
|------|-----------|--------|
| `search_dense()` | top-K 선별 + threshold 필터링 동작 검증 | 낮음 |
| `union_candidates()` | Dense ∪ Sparse 합집합의 중복 제거, 출처 추적 | 중간 |
| `search_hybrid()` | min-max 정규화, 가중 합산 점수 계산 | 중간 |
| `search_colbert_rerank()` | MaxSim 리랭킹 순서 검증 | 중간 |
| `format_context()` | 검색 결과 → 텍스트 포맷팅 | 낮음 |

**구현 방법:**
- `VectorStore`에 직접 `RagDocument`를 삽입하는 헬퍼를 만들어 embedder 의존성 제거
- 미리 계산된 dense/sparse 벡터로 테스트 문서 구성
- 알려진 정답과 비교하여 검색 순서 및 점수 검증

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn make_doc(id: &str, dense: Vec<f32>, sparse: HashMap<u32, f32>) -> RagDocument {
        RagDocument { id: id.into(), text: format!("text_{id}"), dense, sparse }
    }

    fn store_with_docs(docs: Vec<RagDocument>) -> VectorStore {
        VectorStore { documents: docs }
    }

    #[test]
    fn test_search_dense_respects_threshold() { /* ... */ }

    #[test]
    fn test_search_dense_top_k_limit() { /* ... */ }

    #[test]
    fn test_union_deduplicates_candidates() { /* ... */ }

    #[test]
    fn test_hybrid_normalization_single_candidate() { /* ... */ }
}
```

**예상 효과:** 검색 전략 변경 시 회귀 방지. 현재는 전략 파라미터를 바꿀 때마다 수동으로 검증해야 함.

---

### 2. [최우선] 토큰 추정 및 통계 계산

**`estimate_tokens()` (`src/experiment/runner.rs:391`)**

CJK + ASCII 혼합 텍스트의 토큰 수 추정 로직. 벤치마크 지표의 정확도에 직접 영향.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tokens_ascii_only() {
        // "hello world" → 2 words × 1.3 = 2
        assert_eq!(estimate_tokens("hello world"), 2);
    }

    #[test]
    fn test_estimate_tokens_cjk_only() {
        // "안녕하세요" → 5 CJK chars
        assert_eq!(estimate_tokens("안녕하세요"), 5);
    }

    #[test]
    fn test_estimate_tokens_mixed() {
        // CJK + ASCII 혼합
        assert_eq!(estimate_tokens("안녕 hello world"), 2 + 2); // 2 CJK + 2 ASCII words × 1.3
    }

    #[test]
    fn test_estimate_tokens_empty() {
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn test_calc_tps_zero_latency() {
        assert_eq!(calc_tps(100, 0), 0.0);
    }
}
```

**`ExperimentStore::finalize()` (`src/experiment/store.rs:113`)**

실험 요약 통계 (평균/최소/최대 레이턴시, TPS) 계산. 결과 리포팅의 핵심.

테스트 내용:
- 성공한 run만 통계에 포함되는지
- 전부 실패했을 때 0값 반환
- 빈 runs에서의 동작
- min/max/avg 계산 정확성

---

### 3. [높음] TOML 설정 파싱 및 검증

**`ServerProfile::load()` / `validate()` (`src/profile.rs`)**

| 테스트 | 내용 |
|--------|------|
| 유효한 프로필 파싱 | 모든 12개 섹션이 올바르게 로드되는지 |
| 필수 필드 누락 | 적절한 에러 메시지 반환 |
| `validate()` 모델 파일 없음 | bail 에러 확인 |
| `is_speculative()` | speculation 유무 판별 |
| `api_base_url()` | 포트 기반 URL 생성 |

**`Feature::load()` (`src/feature.rs`)**

- 5가지 feature type 각각의 파싱 검증
- 잘못된 type 값에서의 에러 처리

**`PromptSet::load()` (`src/experiment/prompt_set.rs`)**

- `build_system_prompt()` 문자열 조합
- `total_runs()` = prompts.len() × repeat 계산

**구현 방법:** `config/` 디렉토리의 실제 TOML 파일을 사용하거나, 테스트 전용 fixture TOML 작성.

---

### 4. [높음] 직렬화 라운드트립

**`ExperimentStore` JSON/JSONL 직렬화**

```rust
#[test]
fn test_run_record_roundtrip() {
    let record = RunRecord { /* ... */ };
    let json = serde_json::to_string(&record).unwrap();
    let parsed: RunRecord = serde_json::from_str(&json).unwrap();
    assert_eq!(record.prompt_id, parsed.prompt_id);
    // ...
}
```

- `ExperimentMeta` JSON 라운드트립
- `RunRecord` JSONL 라운드트립
- `extra` 필드 `None`일 때 skip_serializing 동작
- `SearchStrategy` serde 태그 기반 역직렬화 (3가지 variant)

---

### 5. [중간] 서버 로그 필터링

**`ExperimentStore::filter_server_log()` (`src/experiment/store.rs:151`)**

```rust
#[test]
fn test_filter_server_log_removes_debug() {
    // ANSI color D 포함 줄 제거 확인
}

#[test]
fn test_filter_server_log_missing_file() {
    // 파일 없으면 Ok(None) 반환
}
```

이 함수는 순수한 텍스트 처리로, 외부 의존성 없이 쉽게 테스트 가능.

---

### 6. [중간] 도구 구현체 (`src/tools/`)

**`LookupKungFu`, `CheckRelationship`, `CheckInventory`**

각 도구의 `call()` 함수는 하드코딩된 데이터 조회로, 테스트가 간단:
- 존재하는 키 → 올바른 결과 반환
- 존재하지 않는 키 → 적절한 에러/빈 결과

복잡도는 낮지만, 도구 데이터 변경 시 회귀 방지에 유용.

---

### 7. [낮음] 서버 인자 빌드 (`src/server.rs`)

**`build_server_args()`**

프로필 설정을 `ServerArgs`로 변환하는 로직. `lmcpp` 외부 의존성 때문에 직접 테스트가 어렵지만, 빌더 로직만 분리하면 테스트 가능:

- GPU 레이어 설정 매핑
- speculation 활성화 시 draft 모델 인자 추가
- `jinja = true` → `--jinja` 플래그 변환
- `parse_reasoning_format()` enum 변환

---

## 테스트 인프라 구축 제안

### 필요 작업

1. **`Cargo.toml`에 dev-dependencies 추가:**
   ```toml
   [dev-dependencies]
   tempfile = "3"        # 임시 파일/디렉토리 (store 테스트용)
   assert_approx_eq = "1" # 부동소수점 비교 (RAG 점수 검증)
   ```

2. **테스트 가능성 개선을 위한 리팩토링:**
   - `VectorStore`에 embedder 없이 직접 문서 삽입하는 `add_raw()` 메서드 추가
   - `estimate_tokens()`, `calc_tps()`를 `pub(crate)`로 변경하거나 같은 모듈 내 테스트 작성
   - `ExperimentStore` 생성자에 base path 주입 가능하게 (tempdir 사용)

3. **테스트 실행 명령:**
   ```bash
   cargo test
   cargo test -- --nocapture  # 로그 출력 포함
   ```

### 권장 커버리지 목표

| 단계 | 대상 | 목표 |
|------|------|------|
| 1단계 | 순수 함수 (estimate_tokens, calc_tps, format_context, filter_server_log) | 100% |
| 2단계 | RAG 검색 알고리즘 | 분기 커버리지 90%+ |
| 3단계 | 설정 파싱 + 직렬화 | 정상 경로 + 에러 경로 |
| 4단계 | 도구 구현체 | 모든 조회 키 커버 |

---

## 요약

| 우선순위 | 영역 | 이유 |
|---------|------|------|
| 최우선 | RAG 검색 알고리즘 | 가장 복잡, 알고리즘 정확성 미검증 |
| 최우선 | 토큰 추정/통계 계산 | 벤치마크 지표에 직접 영향 |
| 높음 | TOML 설정 파싱 | 설정 오류 시 런타임 실패 |
| 높음 | 직렬화 라운드트립 | 데이터 무결성 보장 |
| 중간 | 로그 필터링 | 쉽게 테스트 가능, 텍스트 처리 |
| 중간 | 도구 구현체 | 데이터 변경 시 회귀 방지 |
| 낮음 | 서버 인자 빌드 | 외부 의존성으로 분리 필요 |
