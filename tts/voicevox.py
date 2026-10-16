import re
import asyncio
import logging
from typing import Optional, TYPE_CHECKING, Any

from config import TRANSLATE
from tts.client import TTS
from tts.translation import translate_to_japanese
from utils.file_cleanup import clean_old_audio_files

# 僅在型別檢查時匯入，避免執行時期循環引用
if TYPE_CHECKING:
    from app import MLXChatModel

# 取得應用程式共用的 Logger
logger = logging.getLogger("app")

# 初始化 TTS Client (Speaker ID 58: 猫使ビィ - 人見知り)
tts = TTS(speaker_id=58)

async def synthesize_with_translation(
    text: str, 
    chat_model: Optional[Any] = None 
) -> Optional[str]:
    """
    整合翻譯與語音合成的非同步流程。
    
    流程：
    1. 清洗文字 (移除情緒標籤)。
    2. (可選) 透過本機 MLX 模型翻譯成日文 (Async / ThreadPool)。
    3. 呼叫 Voicevox 生成語音 (Async/HTTPX)。
    4. 背景觸發舊檔案清理 (Thread)。

    Args:
        text (str): 原始回應文字 (可能包含 [emotion:xxx])。
        chat_model (MLXChatModel, optional): 用於翻譯的本機 MLX 模型實例。

    Returns:
        Optional[str]: 生成的音檔 URL (相對路徑)，失敗則回傳 None。
    """
    
    # 1. 移除情緒標籤，只保留要唸出來的文字
    clean_text = re.sub(r"\[emotion:\w+\]", "", text).strip()
    
    if not clean_text:
        logger.warning("[TTS] 文字為空 (可能僅包含情緒標籤)，略過生成")
        return None
        
    logger.info(f"[TTS] 原始文字: {clean_text}")
    
    # 2. 執行翻譯 (非同步)
    if TRANSLATE and chat_model:
        try:
            # translate_to_japanese 內部已封裝 asyncio.to_thread 執行 MLX 推論
            translated_text = await translate_to_japanese(
                clean_text, 
                role_style="可愛、撒嬌語氣", 
                chat_model=chat_model
            )
            logger.info(f"[TTS] 翻譯後文字: {translated_text}")
            clean_text = translated_text
        except Exception as e:
            logger.error(f"[TTS] 翻譯過程發生錯誤，將使用原文生成: {e}")

    # 3. 執行語音合成 (非同步 HTTPX)
    try:
        url = await tts.synthesize_to_file(clean_text)
    except Exception as e:
        logger.error(f"[TTS] Voicevox 生成失敗: {e}")
        return None

    # 4. 背景清理舊語音檔
    asyncio.create_task(asyncio.to_thread(clean_old_audio_files, max_files=20)) 

    return url