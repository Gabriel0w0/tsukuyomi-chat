import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app import MLXChatModel


async def translate_to_japanese(
    text: str, 
    role_style: str = "可愛女僕、撒嬌語氣", 
    chat_model: Any = None 
) -> str:
    if not text or not text.strip():
        return ""
    if chat_model is None:
        return text 

    prompt = f"""請將以下中文台詞翻譯成日文。
要求：
- 保持角色風格：{role_style}
- 第一人稱自稱使用「私 (わたし)」或「月読」，稱呼對方為「ご主人様」
- 不要加入任何動作描寫、括號或多餘解釋，只輸出日文台詞本身

中文台詞：
「{text}」"""

    # 顯式傳入 system，防止 tokenizer 自動塞入 Mistral 預設的長英文
    messages = [
        {"role": "system", "content": "You are a professional Japanese translator."},
        {"role": "user", "content": prompt}
    ]

    try:
        result_text = chat_model.invoke(messages, max_tokens=256)
        # 去除可能帶有的日文引號
        cleaned = result_text.strip().strip("「」\"'")
        return cleaned
    except Exception as e:
        print(f"[Translation Error] {e}")
        return text

