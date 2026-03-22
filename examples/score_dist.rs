//! 전체 문서 점수 분포 확인용 — LLM 없이 검색만 실행

use bge_m3_onnx_rust::{BgeM3Embedder, cosine_similarity, sparse_dot_product};
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    bge_m3_onnx_rust::init_ort();
    let mut embedder = BgeM3Embedder::new(
        "../models/bge-m3/model_quantized.onnx",
        "../models/bge-m3/tokenizer.json",
    )?;

    // 지식 데이터 로드
    let content = std::fs::read_to_string("config/data/npc-knowledge.toml")?;
    let kf: toml::Value = content.parse()?;
    let docs_arr = kf["documents"].as_array().unwrap();

    let mut docs: Vec<(String, Vec<f32>, HashMap<u32, f32>)> = Vec::new();
    for d in docs_arr {
        let id = d["id"].as_str().unwrap().to_string();
        let text = d["text"].as_str().unwrap().to_string();
        let output = embedder.encode(&text)?;
        docs.push((id, output.dense, output.sparse));
    }
    eprintln!("문서 {}개 임베딩 완료\n", docs.len());

    let queries = vec![
        ("무공질문", "이 대협, 무당검법이 어떤 무공인지 설명해 주시겠소?"),
        ("관계질문", "대협, 옥교룡이라는 아이를 아시오? 어떤 사이시오?"),
        ("적대관계", "벽안호라는 자가 위험하다 들었소. 그자에 대해 아는 것이 있소?"),
        ("소지품", "대협, 지금 가지고 계신 검이 어떤 검이오?"),
        ("복합질문", "옥교룡이 비검술을 익히고 있다 하던데, 그 무공은 얼마나 위험하오? 그리고 그 아이와 대협은 어떤 사이시오?"),
        ("배경질문", "대협은 왜 강호를 떠나려 하시오? 무슨 사연이 있소?"),
    ];

    for (label, query) in &queries {
        let q = embedder.encode(query)?;
        let mut scores: Vec<(&str, f32, f32)> = docs.iter()
            .map(|(id, dense, sparse)| {
                let ds = cosine_similarity(&q.dense, dense);
                let ss = sparse_dot_product(&q.sparse, sparse);
                (id.as_str(), ds, ss)
            })
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("=== {} ===", label);
        println!("{:<4} {:<25} {:>8} {:>10}", "순위", "문서ID", "Dense", "Sparse");
        for (i, (id, ds, ss)) in scores.iter().enumerate() {
            let marker = if i < 3 { ">>>" } else if *ds >= 0.4 { "  ." } else { "" };
            println!("{:<4} {:<25} {:>8.3} {:>10.4} {}", i+1, id, ds, ss, marker);
        }
        // 점수 갭 분석
        let top3_avg: f32 = scores[..3].iter().map(|s| s.1).sum::<f32>() / 3.0;
        let rest_avg: f32 = scores[3..].iter().map(|s| s.1).sum::<f32>() / (scores.len() - 3) as f32;
        println!("top3 평균: {:.3}, 나머지 평균: {:.3}, 갭: {:.3}\n",
            top3_avg, rest_avg, top3_avg - rest_avg);
    }
    Ok(())
}
