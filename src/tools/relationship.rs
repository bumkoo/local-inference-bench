use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

/// 캐릭터 간 관계 조회 도구
/// NPC가 다른 캐릭터와의 관계를 확인할 때 사용
#[derive(Debug, Deserialize, Serialize)]
pub struct CheckRelationship;

#[derive(Deserialize)]
pub struct CheckRelationshipArgs {
    pub target: String,
}

#[derive(Debug, thiserror::Error)]
#[error("관계 조회 오류")]
pub struct RelationshipError;

impl Tool for CheckRelationship {
    const NAME: &'static str = "check_relationship";
    type Error = RelationshipError;
    type Args = CheckRelationshipArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        serde_json::from_value(json!({
            "name": "check_relationship",
            "description": "다른 캐릭터와의 관계 정보를 조회합니다. 관계, 호감도, 현재 상태 등을 반환합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "관계를 확인할 대상 캐릭터 이름"
                    }
                },
                "required": ["target"]
            }
        }))
        .expect("Tool Definition")
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        tracing::info!("[tool-call] 관계 조회: {}", args.target);

        let result = match args.target.as_str() {
            s if s.contains("옥교룡") => {
                "옥교룡(玉嬌龍): 관계=제자(비공식), 호감도=60/100, \
                 상태=반항적, 비검술을 독학 중. \
                 이모백이 가르치려 하나 거부하는 상황. \
                 최근 청명검을 몰래 가져간 전적 있음."
            }
            s if s.contains("유수련") => {
                "유수련(俞秀蓮): 관계=연인(미고백), 호감도=95/100, \
                 상태=서로 마음을 알지만 예법에 묶여 고백하지 못함. \
                 맹세형(孟世珩)의 약혼자였으나 사별. \
                 강호 표국(鏢局)을 운영하는 여협."
            }
            s if s.contains("벽안호") || s.contains("장") => {
                "벽안호 장(碧眼狐 張): 관계=적대, 호감도=5/100, \
                 상태=위험 인물. 옥교룡의 스승(암흑면). \
                 독문암기와 기문둔갑에 능함. \
                 과거 이모백의 스승을 암살한 원수."
            }
            _ => {
                return Ok(format!(
                    "'{}' — 해당 인물과의 관계 기록이 없습니다. 강호에서 아직 인연이 닿지 않은 자입니다.",
                    args.target
                ));
            }
        };

        Ok(result.to_string())
    }
}
