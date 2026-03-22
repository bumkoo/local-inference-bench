use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

/// 캐릭터 소지품 조회 도구
/// NPC가 자신 또는 다른 캐릭터의 소지품을 확인할 때 사용
#[derive(Debug, Deserialize, Serialize)]
pub struct CheckInventory;

#[derive(Deserialize)]
pub struct CheckInventoryArgs {
    pub character: String,
}

#[derive(Debug, thiserror::Error)]
#[error("소지품 조회 오류")]
pub struct InventoryError;

impl Tool for CheckInventory {
    const NAME: &'static str = "check_inventory";
    type Error = InventoryError;
    type Args = CheckInventoryArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        serde_json::from_value(json!({
            "name": "check_inventory",
            "description": "캐릭터의 소지품 목록을 조회합니다. 무기, 약품, 중요 물품 등을 반환합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "character": {
                        "type": "string",
                        "description": "소지품을 확인할 캐릭터 이름"
                    }
                },
                "required": ["character"]
            }
        }))
        .expect("Tool Definition")
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let start = std::time::Instant::now();
        tracing::info!("[tool-call] 소지품 조회 요청: character={:?}", args.character);

        let result = match args.character.as_str() {
            s if s.contains("이모백") => {
                "이모백 소지품: \
                 [무기] 청명검(靑冥劍) — 400년 된 명검, 현재 패용 중. \
                 [문서] 무당파 장문인 신표. \
                 [약품] 해독단 2개, 금창약 1개. \
                 [기타] 서신 1통 (유수련에게 보내지 못한 편지)."
            }
            s if s.contains("옥교룡") => {
                "옥교룡 소지품: \
                 [무기] 단검 1자루 (평범한 물건). \
                 [문서] 벽안호에게 받은 비검술 비급 (일부 해독). \
                 [약품] 없음. \
                 [기타] 옥비녀 (어머니의 유품), 은냥 30량."
            }
            s if s.contains("진씨") || s.contains("약방") => {
                "진씨 약방 재고: \
                 [약재] 천산설련 1주 (매우 귀중), 녹용 3근, 인삼 5근. \
                 [완제] 해독단 10개, 지혈산 20포, 회춘환 3알. \
                 [기타] 약방 장부, 은냥 150량."
            }
            _ => {
                let not_found = format!(
                    "'{}' — 해당 인물의 소지품 정보가 없습니다.",
                    args.character
                );
                tracing::info!("[tool-result] 소지품 조회 완료: character={:?}, {}ms, 미발견",
                    args.character, start.elapsed().as_millis());
                return Ok(not_found);
            }
        };

        tracing::info!("[tool-result] 소지품 조회 완료: character={:?}, {}ms, {}bytes",
            args.character, start.elapsed().as_millis(), result.len());
        Ok(result.to_string())
    }
}
