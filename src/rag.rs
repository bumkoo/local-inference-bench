//! RAG 모듈 — BGE-M3 3종 벡터 + Union 기반 전략 검색
//!
//! ## 검색 전략
//! - **DenseOnly**: Dense top-K → threshold 필터
//! - **Hybrid**: Dense top-K(threshold) ∪ Sparse top-K/2 → 가중 합산 리랭킹 → top-K
//! - **ColBERTRerank**: Dense top-K(threshold) ∪ Sparse top-K/2 → ColBERT 리랭킹 → top-K
//!
//! ## Union 설계 원칙
//! - Dense: 의미 검색의 주축. top-K 선별 후 threshold로 노이즈 제거.
//! - Sparse: 키워드 보조. threshold 없이 top-K/2만 선별 (정답이 Sparse 0일 수 있음).
//! - 합집합에 대해서만 리랭킹 수행 → 벡터 DB(HNSW + 역인덱스) 전환 시 그대로 매핑.

use bge_m3_onnx_rust::{
    BgeM3Embedder, BgeM3Output,
    cosine_similarity, sparse_dot_product, max_sim,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use tracing::info;

// ===== 검색 전략 =====

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SearchStrategy {
    /// 전략 1: Dense top-K → threshold 필터
    #[serde(rename = "dense_only")]
    DenseOnly {
        #[serde(default = "default_top_k")]
        top_k: usize,
        /// Dense 코사인 유사도 threshold (기본 0.38)
        #[serde(default = "default_dense_threshold")]
        threshold: f32,
    },

    /// 전략 2: Dense top-K(threshold) ∪ Sparse top-K/2 → 가중 합산 리랭킹 → top-K
    #[serde(rename = "hybrid")]
    Hybrid {
        /// Dense 인덱스에서 뽑을 후보 수 (Sparse는 자동으로 절반)
        #[serde(default = "default_candidate_k")]
        candidate_k: usize,
        /// 최종 반환 수
        #[serde(default = "default_top_k")]
        top_k: usize,
        /// Dense threshold (1차 Dense 후보 필터링용, 기본 0.38)
        #[serde(default = "default_dense_threshold")]
        dense_threshold: f32,
        /// Dense 점수 가중치 (기본 0.7)
        #[serde(default = "default_dense_weight")]
        dense_weight: f32,
        /// Sparse 점수 가중치 (기본 0.3)
        #[serde(default = "default_sparse_weight")]
        sparse_weight: f32,
    },

    /// 전략 3: Dense top-K(threshold) ∪ Sparse top-K/2 → ColBERT 리랭킹 → top-K
    #[serde(rename = "colbert_rerank")]
    ColBERTRerank {
        /// Dense 인덱스에서 뽑을 후보 수 (Sparse는 자동으로 절반)
        #[serde(default = "default_candidate_k")]
        candidate_k: usize,
        /// 최종 반환 수
        #[serde(default = "default_top_k")]
        top_k: usize,
        /// Dense threshold (1차 Dense 후보 필터링용, 기본 0.38)
        #[serde(default = "default_dense_threshold")]
        dense_threshold: f32,
    },
}

fn default_top_k() -> usize { 3 }
fn default_dense_threshold() -> f32 { 0.38 }
fn default_dense_weight() -> f32 { 0.7 }
fn default_sparse_weight() -> f32 { 0.3 }
fn default_candidate_k() -> usize { 10 }

impl Default for SearchStrategy {
    fn default() -> Self {
        SearchStrategy::DenseOnly {
            top_k: default_top_k(),
            threshold: default_dense_threshold(),
        }
    }
}

impl SearchStrategy {
    pub fn name(&self) -> &'static str {
        match self {
            SearchStrategy::DenseOnly { .. } => "dense_only",
            SearchStrategy::Hybrid { .. } => "hybrid",
            SearchStrategy::ColBERTRerank { .. } => "colbert_rerank",
        }
    }

    pub fn top_k(&self) -> usize {
        match self {
            SearchStrategy::DenseOnly { top_k, .. } => *top_k,
            SearchStrategy::Hybrid { top_k, .. } => *top_k,
            SearchStrategy::ColBERTRerank { top_k, .. } => *top_k,
        }
    }
}

// ===== 문서 & 벡터 스토어 =====

#[derive(Debug, Clone)]
pub struct RagDocument {
    pub id: String,
    pub text: String,
    pub dense: Vec<f32>,
    pub sparse: HashMap<u32, f32>,
}

pub struct VectorStore {
    documents: Vec<RagDocument>,
}

impl VectorStore {
    pub fn new() -> Self {
        Self { documents: Vec::new() }
    }

    pub fn add(&mut self, id: &str, text: &str, embedder: &mut BgeM3Embedder) -> anyhow::Result<()> {
        let output = embedder.encode(text)?;
        self.documents.push(RagDocument {
            id: id.to_string(),
            text: text.to_string(),
            dense: output.dense,
            sparse: output.sparse,
        });
        Ok(())
    }

    pub fn add_batch(
        &mut self,
        docs: &[(String, String)],
        embedder: &mut BgeM3Embedder,
    ) -> anyhow::Result<()> {
        info!("벡터 스토어: {}개 문서 임베딩 시작 (Dense+Sparse)", docs.len());
        for (id, text) in docs {
            self.add(id, text, embedder)?;
        }
        info!("벡터 스토어: {}개 문서 임베딩 완료 (총 {}개)", docs.len(), self.documents.len());
        Ok(())
    }

    /// 전략 기반 검색
    pub fn search(
        &self,
        query: &BgeM3Output,
        strategy: &SearchStrategy,
        embedder: &mut BgeM3Embedder,
    ) -> Vec<SearchResult> {
        match strategy {
            SearchStrategy::DenseOnly { top_k, threshold } => {
                self.search_dense(&query.dense, *top_k, *threshold)
            }
            SearchStrategy::Hybrid {
                candidate_k, top_k, dense_threshold, dense_weight, sparse_weight,
            } => {
                self.search_hybrid(
                    &query.dense, &query.sparse,
                    *candidate_k, *dense_threshold,
                    *dense_weight, *sparse_weight, *top_k,
                )
            }
            SearchStrategy::ColBERTRerank {
                candidate_k, top_k, dense_threshold,
            } => {
                self.search_colbert_rerank(
                    &query.dense, &query.sparse, &query.colbert,
                    *candidate_k, *dense_threshold, *top_k,
                    embedder,
                )
            }
        }
    }

    pub fn query(
        &self,
        query_text: &str,
        embedder: &mut BgeM3Embedder,
        strategy: &SearchStrategy,
    ) -> anyhow::Result<Vec<SearchResult>> {
        let query_output = embedder.encode(query_text)?;
        Ok(self.search(&query_output, strategy, embedder))
    }

    // ── 전략 1: Dense top-K → threshold 필터 ──

    fn search_dense(&self, query_dense: &[f32], top_k: usize, threshold: f32) -> Vec<SearchResult> {
        let mut scored: Vec<SearchResult> = self.documents.iter()
            .map(|doc| {
                let ds = cosine_similarity(query_dense, &doc.dense);
                SearchResult {
                    id: doc.id.clone(),
                    text: doc.text.clone(),
                    score: ds,
                    detail: ScoreDetail::Dense { dense_score: ds },
                }
            })
            .collect();
        // top-K 먼저
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        // threshold 후적용
        scored.retain(|r| r.score >= threshold);
        scored
    }

    // ── 공통: Dense top-K(threshold) ∪ Sparse top-K/2 합집합 ──
    // Dense: 의미 검색 주축, threshold로 노이즈 제거
    // Sparse: 키워드 보조, threshold 없이 절반만 (정답이 Sparse 0일 수 있으므로)

    fn union_candidates(
        &self,
        query_dense: &[f32],
        query_sparse: &HashMap<u32, f32>,
        candidate_k: usize,
        dense_threshold: f32,
    ) -> Vec<CandidateDoc> {
        let sparse_k = candidate_k / 2;

        // Dense top-K → threshold 필터
        let mut dense_scored: Vec<(usize, f32)> = self.documents.iter()
            .enumerate()
            .map(|(i, doc)| (i, cosine_similarity(query_dense, &doc.dense)))
            .collect();
        dense_scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        dense_scored.truncate(candidate_k);
        dense_scored.retain(|&(_, score)| score >= dense_threshold);

        // Sparse top-K/2 (threshold 없음)
        let mut sparse_scored: Vec<(usize, f32)> = self.documents.iter()
            .enumerate()
            .map(|(i, doc)| (i, sparse_dot_product(query_sparse, &doc.sparse)))
            .collect();
        sparse_scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sparse_scored.truncate(sparse_k);

        // 합집합 (중복 제거, 출처 추적)
        let dense_ids: HashSet<usize> = dense_scored.iter().map(|&(i, _)| i).collect();
        let sparse_ids: HashSet<usize> = sparse_scored.iter().map(|&(i, _)| i).collect();
        let dense_map: HashMap<usize, f32> = dense_scored.into_iter().collect();
        let sparse_map: HashMap<usize, f32> = sparse_scored.into_iter().collect();

        let mut seen = HashSet::new();
        let mut candidates = Vec::new();

        for &idx in dense_ids.union(&sparse_ids) {
            if seen.insert(idx) {
                let ds = dense_map.get(&idx).copied()
                    .unwrap_or_else(|| cosine_similarity(query_dense, &self.documents[idx].dense));
                let ss = sparse_map.get(&idx).copied()
                    .unwrap_or_else(|| sparse_dot_product(query_sparse, &self.documents[idx].sparse));
                candidates.push(CandidateDoc {
                    idx, dense_score: ds, sparse_score_raw: ss,
                    from_dense: dense_ids.contains(&idx),
                    from_sparse: sparse_ids.contains(&idx),
                });
            }
        }

        info!("[union] Dense {}개(threshold≥{:.2}) + Sparse {}개(top-{}) → 합집합 {}개",
            dense_ids.len(), dense_threshold, sparse_ids.len(), sparse_k, candidates.len());
        candidates
    }

    // ── 전략 2: Hybrid (Union → 가중 합산 리랭킹 → top-K) ──

    fn search_hybrid(
        &self,
        query_dense: &[f32],
        query_sparse: &HashMap<u32, f32>,
        candidate_k: usize,
        dense_threshold: f32,
        dense_weight: f32,
        sparse_weight: f32,
        top_k: usize,
    ) -> Vec<SearchResult> {
        let candidates = self.union_candidates(query_dense, query_sparse, candidate_k, dense_threshold);

        // 합집합 내에서 Sparse min-max 정규화
        let sp_max = candidates.iter().map(|c| c.sparse_score_raw).fold(f32::NEG_INFINITY, f32::max);
        let sp_min = candidates.iter().map(|c| c.sparse_score_raw).fold(f32::INFINITY, f32::min);
        let sp_range = sp_max - sp_min;

        let mut scored: Vec<SearchResult> = candidates.iter()
            .map(|c| {
                let ss_norm = if sp_range > 1e-9 {
                    (c.sparse_score_raw - sp_min) / sp_range
                } else { 0.0 };
                let combined = dense_weight * c.dense_score + sparse_weight * ss_norm;
                SearchResult {
                    id: self.documents[c.idx].id.clone(),
                    text: self.documents[c.idx].text.clone(),
                    score: combined,
                    detail: ScoreDetail::Hybrid {
                        dense_score: c.dense_score,
                        sparse_score_raw: c.sparse_score_raw,
                        sparse_score_norm: ss_norm,
                        dense_weight, sparse_weight,
                        from_dense: c.from_dense,
                        from_sparse: c.from_sparse,
                    },
                }
            })
            .collect();
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    // ── 전략 3: ColBERT Rerank (Union → 후보 재인코딩 → MaxSim → top-K) ──

    fn search_colbert_rerank(
        &self,
        query_dense: &[f32],
        query_sparse: &HashMap<u32, f32>,
        query_colbert: &[Vec<f32>],
        candidate_k: usize,
        dense_threshold: f32,
        top_k: usize,
        embedder: &mut BgeM3Embedder,
    ) -> Vec<SearchResult> {
        let candidates = self.union_candidates(query_dense, query_sparse, candidate_k, dense_threshold);

        info!("[colbert] {}개 후보 문서 재인코딩 시작", candidates.len());
        let mut reranked: Vec<SearchResult> = candidates.iter()
            .filter_map(|c| {
                let doc = &self.documents[c.idx];
                let doc_output = embedder.encode(&doc.text).ok()?;
                let colbert_score = max_sim(query_colbert, &doc_output.colbert);
                Some(SearchResult {
                    id: doc.id.clone(),
                    text: doc.text.clone(),
                    score: colbert_score,
                    detail: ScoreDetail::ColBERTRerank {
                        dense_score: c.dense_score,
                        sparse_score_raw: c.sparse_score_raw,
                        colbert_score,
                        from_dense: c.from_dense,
                        from_sparse: c.from_sparse,
                    },
                })
            })
            .collect();
        reranked.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        reranked.truncate(top_k);
        reranked
    }

    pub fn len(&self) -> usize { self.documents.len() }
    pub fn is_empty(&self) -> bool { self.documents.is_empty() }
}

// ===== 내부 구조체 =====

struct CandidateDoc {
    idx: usize,
    dense_score: f32,
    sparse_score_raw: f32,
    from_dense: bool,
    from_sparse: bool,
}

// ===== 검색 결과 =====

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub text: String,
    pub score: f32,
    pub detail: ScoreDetail,
}

#[derive(Debug, Clone, Serialize)]
pub enum ScoreDetail {
    Dense {
        dense_score: f32,
    },
    Hybrid {
        dense_score: f32,
        sparse_score_raw: f32,
        sparse_score_norm: f32,
        dense_weight: f32,
        sparse_weight: f32,
        from_dense: bool,
        from_sparse: bool,
    },
    ColBERTRerank {
        dense_score: f32,
        sparse_score_raw: f32,
        colbert_score: f32,
        from_dense: bool,
        from_sparse: bool,
    },
}

pub fn format_context(results: &[SearchResult]) -> String {
    results.iter()
        .enumerate()
        .map(|(i, r)| format!("[참고{}] (점수:{:.3})\n{}", i + 1, r.score, r.text))
        .collect::<Vec<_>>()
        .join("\n\n")
}

// ===== TOML 지식 데이터 로더 =====

#[derive(Debug, Deserialize)]
struct KnowledgeFile {
    documents: Vec<KnowledgeDoc>,
}

#[derive(Debug, Deserialize)]
struct KnowledgeDoc {
    id: String,
    #[allow(dead_code)]
    category: String,
    text: String,
}

pub fn load_knowledge(
    path: &Path,
    embedder: &mut BgeM3Embedder,
) -> anyhow::Result<VectorStore> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("지식 데이터 로드 실패 {:?}: {}", path, e))?;
    let kf: KnowledgeFile = toml::from_str(&content)?;
    let docs: Vec<(String, String)> = kf.documents.iter()
        .map(|d| (d.id.clone(), d.text.clone()))
        .collect();
    let mut store = VectorStore::new();
    store.add_batch(&docs, embedder)?;
    Ok(store)
}
