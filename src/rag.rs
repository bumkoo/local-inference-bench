//! RAG 모듈 — BGE-M3 ONNX 인프로세스 임베딩 + 인메모리 벡터 스토어
//!
//! ## 설계
//! - `bge-m3-onnx-rust`의 `BgeM3Embedder`를 직접 사용 (외부 서버 없음)
//! - Dense 벡터(1024차원)로 코사인 유사도 검색
//! - `.context()` / `.dynamic_context()`의 `<file>` 태그 문제를 회피
//!   → 검색 결과를 preamble에 직접 합침

use bge_m3_onnx_rust::{BgeM3Embedder, cosine_similarity};
use serde::Deserialize;
use std::path::Path;
use tracing::info;

/// 벡터 스토어에 저장되는 문서
#[derive(Debug, Clone)]
pub struct RagDocument {
    pub id: String,
    pub text: String,
    pub dense: Vec<f32>,  // [1024] BGE-M3 Dense 벡터
}

/// 인메모리 벡터 스토어
pub struct VectorStore {
    documents: Vec<RagDocument>,
}

impl VectorStore {
    pub fn new() -> Self {
        Self { documents: Vec::new() }
    }

    /// 텍스트 문서를 임베딩하여 스토어에 추가
    pub fn add(&mut self, id: &str, text: &str, embedder: &mut BgeM3Embedder) -> anyhow::Result<()> {
        let dense = embedder.encode_dense(text)?;
        self.documents.push(RagDocument {
            id: id.to_string(),
            text: text.to_string(),
            dense,
        });
        Ok(())
    }

    /// 여러 문서를 한 번에 추가
    pub fn add_batch(
        &mut self,
        docs: &[(String, String)],  // (id, text) 쌍
        embedder: &mut BgeM3Embedder,
    ) -> anyhow::Result<()> {
        info!("벡터 스토어: {}개 문서 임베딩 시작", docs.len());
        for (id, text) in docs {
            self.add(id, text, embedder)?;
        }
        info!("벡터 스토어: {}개 문서 임베딩 완료 (총 {}개)", docs.len(), self.documents.len());
        Ok(())
    }

    /// Dense 코사인 유사도로 상위 N개 검색
    pub fn search(&self, query_dense: &[f32], top_k: usize) -> Vec<SearchResult> {
        let mut scored: Vec<SearchResult> = self.documents.iter()
            .map(|doc| SearchResult {
                id: doc.id.clone(),
                text: doc.text.clone(),
                score: cosine_similarity(query_dense, &doc.dense),
            })
            .collect();
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    /// 텍스트 쿼리로 검색 (임베딩 + 검색 한번에)
    pub fn query(&self, query_text: &str, embedder: &mut BgeM3Embedder, top_k: usize) -> anyhow::Result<Vec<SearchResult>> {
        let query_dense = embedder.encode_dense(query_text)?;
        Ok(self.search(&query_dense, top_k))
    }

    /// 문서 수
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }
}

/// 검색 결과
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub text: String,
    pub score: f32,
}

/// 검색 결과를 preamble에 합칠 수 있는 텍스트로 포맷
pub fn format_context(results: &[SearchResult]) -> String {
    results.iter()
        .enumerate()
        .map(|(i, r)| format!("[참고{}] (유사도:{:.3})\n{}", i + 1, r.score, r.text))
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

/// TOML 파일에서 지식 데이터를 로드하여 벡터 스토어에 적재
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
