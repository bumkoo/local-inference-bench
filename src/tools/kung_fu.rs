use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

/// 무공 정보 조회 도구
/// NPC가 무공에 대해 대화할 때 정확한 정보를 참조
#[derive(Debug, Deserialize, Serialize)]
pub struct LookupKungFu;

#[derive(Deserialize)]
pub struct LookupKungFuArgs {
    pub name: String,
}

#[derive(Debug, thiserror::Error)]
#[error("무공 조회 오류")]
pub struct KungFuError;

impl Tool for LookupKungFu {
    const NAME: &'static str = "lookup_kung_fu";
    type Error = KungFuError;
    type Args = LookupKungFuArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        serde_json::from_value(json!({
            "name": "lookup_kung_fu",
            "description": "무공(武功)의 상세 정보를 조회합니다. 무공명을 입력하면 유파, 특징, 위력 등을 반환합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "조회할 무공의 이름 (예: 무당검법, 비검술, 구양진경)"
                    }
                },
                "required": ["name"]
            }
        }))
        .expect("Tool Definition")
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        tracing::info!("[tool-call] 무공 조회: {}", args.name);

        // 게임 데이터 시뮬레이션 — 나중에 실제 DB/JSON으로 교체
        let result = match args.name.as_str() {
            s if s.contains("무당검법") || s.contains("武當劍法") => {
                "무당검법(武當劍法): 무당파 정통 검술. \
                 이유극강(以柔克剛) — 부드러움으로 강함을 제압하는 원리. \
                 내공 수련이 깊을수록 검의 위력이 배가됨. \
                 대표 초식: 신문칠식(神門七式), 태극검법. \
                 약점: 초기 전개가 느려 급습에 취약.".to_string()
            }
            s if s.contains("비검술") || s.contains("飛劍術") => {
                "비검술(飛劍術): 청성파 계열 암기 검술. \
                 검을 손에서 놓아 원거리 공격. 정확도 극상. \
                 미완성 시 내력 소모가 극심하여 연속 사용 불가. \
                 완성 시 백보천양(百步穿楊) 경지에 도달.".to_string()
            }
            s if s.contains("구양진경") || s.contains("九陽真經") => {
                "구양진경(九陽真經): 전설의 내공 심법. \
                 순양(純陽)의 내력을 극대화. \
                 한서불침(寒暑不侵) — 추위와 더위에 영향받지 않음. \
                 모든 무공의 기초 내공으로 활용 가능.".to_string()
            }
            s if s.contains("태극권") || s.contains("太極拳") => {
                "태극권(太極拳): 무당파 권법. \
                 사량발천근(四兩撥千斤) — 작은 힘으로 큰 힘을 흘려보냄. \
                 공수일체(攻守一體). 방어와 공격이 하나. \
                 노년의 고수일수록 위력이 강함.".to_string()
            }
            _ => {
                format!("'{}' — 해당 무공에 대한 기록을 찾을 수 없습니다. \
                 강호에 알려지지 않은 비전(秘傳)일 수 있습니다.", args.name)
            }
        };

        Ok(result)
    }
}
