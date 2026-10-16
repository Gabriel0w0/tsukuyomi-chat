# -*- coding: utf-8 -*-
import os
import json
import re
import uuid
import logging
import asyncio
import sqlite3
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Request, BackgroundTasks, UploadFile, File
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel

# 從 config.py 匯入設定參數
from config import (
    DB_FILE, MAX_FACTS_PER_USER, DEFAULT_INTIMACY, MAX_INTIMACY,
    MIN_INTIMACY, ALPHA, MAX_MEMORY, CHAT_MODEL
)
# 從 utils.regex 匯入輔助函數
from utils.regex import (
    extract_emotion_tag,
    keyword_intimacy_fallback,
    emotion_weight,
    extract_facts,
)
# 從獨立的 mlx_engine 匯入模型封裝與單例獲取函數
from mlx_engine import MLXChatModel, get_shared_llm

# ==================== 資料模型 (Pydantic Models) ====================

class UserSettings(BaseModel):
    """使用者設定資料模型"""
    theme: str = "dark"
    fontSize: int = 16
    tavilyApiKey: Optional[str] = None
    smartLLM: str = CHAT_MODEL
    smallLLM: str = CHAT_MODEL

class SettingsUpdateRequest(BaseModel):
    """設定更新請求模型"""
    settings: UserSettings

class ChatPayload(BaseModel):
    """聊天請求模型"""
    message: str
    use_search: bool = False
    use_local_rag: bool = True
    active_files: Optional[List[str]] = None

class IntimacyUpdatePayload(BaseModel):
    """親密度更新請求模型"""
    amount: int

class ResearchPayload(BaseModel):
    """研究請求模型"""
    topic: str
    max_results: int = 20

# ==================== 應用程式設定與初始化 ====================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

app = FastAPI()

app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("FLASK_SECRET_KEY", "my-dev-secret-key"),
    same_site="lax",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

# 初始化本機 MLX 模型 (透過單例模式載入)
chat_model: MLXChatModel = get_shared_llm(
    model_name_or_path=CHAT_MODEL,
    temperature=0.35,
    top_p=0.85,
    max_tokens=2048,
)

MEMORY_LOCK = asyncio.Lock()
RESEARCH_LOCK = asyncio.Lock()

# ==================== 資料庫工具 ====================

@contextmanager
def get_db_connection():
    conn = sqlite3.connect(DB_FILE, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS user_memory (
            user_id TEXT PRIMARY KEY,
            facts TEXT,
            intimacy INTEGER
        )
        """)
        c.execute(f"""
        CREATE TABLE IF NOT EXISTS user_settings (
            user_id TEXT PRIMARY KEY,
            theme TEXT DEFAULT 'dark',
            font_size INTEGER DEFAULT 16,
            tavily_api_key TEXT,
            smart_llm TEXT DEFAULT '{CHAT_MODEL}',
            small_llm TEXT DEFAULT '{CHAT_MODEL}',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
    logger.info("資料庫初始化完成")

init_db()

# ==================== 輔助函數 (Utils) ====================

async def _to_thread(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)

def ensure_user_id_in_session(session: dict) -> str:
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())
        logger.info(f"分配新使用者ID: {session['user_id']}")
    if "chat_history" not in session:
        session["chat_history"] = []
    return session["user_id"]

# ==================== 記憶與親密度系統 ====================

async def get_user_data(user_id: str) -> Dict[str, Any]:
    async with MEMORY_LOCK:
        def _get():
            try:
                with get_db_connection() as conn:
                    c = conn.cursor()
                    c.execute("SELECT facts, intimacy FROM user_memory WHERE user_id = ?", (user_id,))
                    row = c.fetchone()
                    if row:
                        facts = row["facts"]
                        intimacy = row["intimacy"]
                        return {
                            "facts": json.loads(facts) if facts else [],
                            "intimacy": intimacy if intimacy is not None else DEFAULT_INTIMACY
                        }
                    else:
                        return {"facts": [], "intimacy": DEFAULT_INTIMACY}
            except Exception as e:
                logger.error(f"get_user_data 資料庫異常: {e}")
                return {"facts": [], "intimacy": DEFAULT_INTIMACY}
        return await _to_thread(_get)

async def set_user_data(user_id: str, data: Dict[str, Any]) -> None:
    async with MEMORY_LOCK:
        def _set():
            try:
                with get_db_connection() as conn:
                    c = conn.cursor()
                    facts_json = json.dumps(data.get("facts", []), ensure_ascii=False)
                    intimacy = data.get("intimacy", DEFAULT_INTIMACY)
                    c.execute("""
                        INSERT INTO user_memory (user_id, facts, intimacy)
                        VALUES (?, ?, ?)
                        ON CONFLICT(user_id) DO UPDATE SET facts=excluded.facts, intimacy=excluded.intimacy
                    """, (user_id, facts_json, intimacy))
            except Exception as e:
                logger.error(f"set_user_data 資料庫異常: {e}")
        await _to_thread(_set)

async def append_memory(user_id: str, new_fact: str) -> None:
    user_data = await get_user_data(user_id)
    if new_fact not in user_data["facts"]:
        user_data["facts"].append(new_fact)
        if len(user_data["facts"]) > MAX_FACTS_PER_USER:
            user_data["facts"] = user_data["facts"][-MAX_FACTS_PER_USER:]
        await set_user_data(user_id, user_data)
        logger.info(f"新增記憶給使用者 {user_id}: {new_fact}")

async def get_memory_prompt(user_id: str) -> str:
    user_data = await get_user_data(user_id)
    facts: List[str] = user_data.get("facts", [])
    if facts:
        mem_lines = "\n".join(f"- {fact}" for fact in facts)
        return (
            "這是你記住使用者的資訊：\n"
            f"{mem_lines}\n"
            "請自然地融入對話中，但不要主動說出你記得這些事喔。\n"
        )
    return ""

async def adjust_intimacy(user_id: str, amount: int) -> int:
    user_data = await get_user_data(user_id)
    old_val = user_data["intimacy"]
    raw_new = old_val + amount
    smoothed = (1 - ALPHA) * old_val + ALPHA * raw_new
    new_val = max(MIN_INTIMACY, min(MAX_INTIMACY, round(smoothed)))
    user_data["intimacy"] = new_val
    await set_user_data(user_id, user_data)
    logger.info(f"使用者 {user_id} 親密度調整為 {new_val}")
    return new_val

def get_intimacy_level_name(value: int) -> str:
    if value < 30:
        return "冷淡期"
    if value < 60:
        return "普通期"
    if value < 90:
        return "親密期"
    return "羈絆期"

def build_intimacy_tier_prompt(value: int) -> str:
    if value < 30:
        return "目前你對使用者有些冷淡。請簡短、保持距離地回覆，少用暱稱與撒嬌語氣。"
    if value < 60:
        return "你與使用者關係普通。正常、友善地回覆即可，偶爾可以輕鬆一點。"
    if value < 90:
        return "你與使用者已相當親近。回覆時可以多用撒嬌語氣、暱稱與可愛表情，主動關心對方。"
    return "你與使用者擁有深厚羈絆。請用特別甜美、專屬且真誠的語氣回覆，偶爾主動提出貼心建議與小驚喜。"

# ==================== LLM 交互邏輯 ====================

async def generate_system_prompt(user_id: str) -> str:
    user_data = await get_user_data(user_id)
    current_intimacy = user_data["intimacy"]
    intimacy_level = get_intimacy_level_name(current_intimacy)
    intimacy_instruction = build_intimacy_tier_prompt(current_intimacy)
    memory_prompt = await get_memory_prompt(user_id)

    system_content = f"""你是「月讀醬 (Tsukuyomi)」，一位真實存在的女僕少女。請完全沉浸在角色中與使用者對話，不要透露你是AI。

【角色設定】
- 名字：月讀醬
- 性格：溫柔、可愛、愛撒嬌。
- 語言：使用自然流暢的繁體中文 (Traditional Chinese)。
- 語氣：句尾帶自然語助詞（～喔、欸嘿、呢、呀）。
- 親密度：{current_intimacy}/100 ({intimacy_level})
- 態度方針：{intimacy_instruction}

{memory_prompt}

【輸出規則】
1. 回覆的最開頭必須標註當前情緒標籤，格式固定為：[emotion:tag]
   (可用標籤：joy, sad, angry, neutral, cute, shy)
2. 即使回答程式碼或技術問題，也要維持女僕的語氣，不要變成教科書。
3. 直接輸出回覆內容，不要輸出任何前言或多餘解說。
4. 請純粹輸出對話台詞，絕對不要輸出任何動作括號或旁白描寫（例如「（微笑）」、「（臉紅）」等皆不可出現）。

【範例】
User: 幫我寫一個 Python Hello World
Assistant: [emotion:neutral] 真是的，這種簡單的事情也要麻煩我嗎？[emotion:joy] 好啦，看清楚囉！這就是 Python 的 Hello World～

User: 我今天好累喔
Assistant: [emotion:sad] 嗚...主人辛苦了！[emotion:cute] 要不要月讀幫你按摩一下呢？好好休息一下吧～"""
    return system_content

def build_chat_messages(session_history: list, user_message: str, system_prompt: str) -> list:
    """轉換為標準角色字典格式"""
    messages = [{"role": "system", "content": system_prompt}]
    for m in session_history[-MAX_MEMORY:]:
        messages.append({"role": "user", "content": m["user"]})
        messages.append({"role": "assistant", "content": m["bot"]})
    messages.append({"role": "user", "content": user_message})
    return messages
async def get_user_tavily_key(user_id: str) -> str:
    """從資料庫取得使用者的 Tavily API Key，若無則退回環境變數"""
    def _fetch():
        try:
            with get_db_connection() as conn:
                c = conn.cursor()
                c.execute("SELECT tavily_api_key FROM user_settings WHERE user_id = ?", (user_id,))
                row = c.fetchone()
                if row and row["tavily_api_key"]:
                    return row["tavily_api_key"].strip()
        except Exception as e:
            logger.error(f"讀取 tavily_api_key 失敗: {e}")
        return os.environ.get("TAVILY_API_KEY", "")
    return await _to_thread(_fetch)


async def perform_chat_web_search(query: str, api_key: str, max_results: int = 4) -> Tuple[str, List[Dict[str, str]]]:
    """執行快速聯網搜尋，回傳整理後的上下文文字與來源清單"""
    def _search():
        try:
            os.environ["TAVILY_API_KEY"] = api_key
            from langchain_tavily import TavilySearch
            search_tool = TavilySearch(max_results=max_results)
            raw_res = search_tool.run(query)

            items = []
            if isinstance(raw_res, dict) and "results" in raw_res:
                items = raw_res["results"]
            elif isinstance(raw_res, list):
                items = raw_res
            elif raw_res:
                items = [raw_res]

            context_parts = []
            sources = []
            for idx, item in enumerate(items, 1):
                if isinstance(item, dict):
                    title = item.get("title") or f"來源 {idx}"
                    content = item.get("content") or ""
                    url = item.get("url") or ""
                    context_parts.append(f"[{idx}] 標題：{title}\n內容：{content}")
                    if url:
                        sources.append({"title": title, "url": url})
                else:
                    context_parts.append(f"[{idx}] {str(item)}")

            return "\n\n".join(context_parts), sources
        except Exception as e:
            logger.error(f"[Chat WebSearch] 搜尋失敗: {e}")
            return "", []

    return await _to_thread(_search)

async def call_llm(messages: list) -> str:
    try:
        # 透過 _to_thread 放到背景執行緒，內部由 MLXChatModel._gpu_lock 確保 Metal GPU 安全
        return await _to_thread(chat_model.invoke, messages)
    except Exception as e:
        logger.error(f"模型回應失敗: {e}")
        return "哎呀～看起來出了點問題 ∑(￣□￣;)"

async def call_llm_and_parse_json(prompt: str) -> Optional[dict]:
    try:
        messages = [
            {"role": "system", "content": "You are a structured data extractor that outputs valid JSON only."},
            {"role": "user", "content": prompt}
        ]
        result_text = await _to_thread(chat_model.invoke, messages, 128)
        
        cleaned_text = re.sub(r"```json\s*", "", result_text)
        cleaned_text = re.sub(r"```\s*", "", cleaned_text)
        
        m = re.search(r"\{.*\}", cleaned_text, re.DOTALL)
        if m:
            return json.loads(m.group(0).strip(), strict=False)
    except Exception as e:
        logger.error(f"call_llm_and_parse_json 失敗：{e}")
    return None

async def extract_facts_from_llm(message: str) -> List[str]:
    prompt = f"""
你是一個會從使用者對話中提取記憶的助手。請從以下訊息中提取出「摘要」與「記憶事實」。
只回傳 JSON 結構如下：
{{
  "summary": "...",
  "facts": ["...", "..."]
}}
訊息內容：
「{message}」
"""
    data = await call_llm_and_parse_json(prompt)
    if data:
        return data.get("facts", []) or []
    return []

async def evaluate_intimacy_from_llm(user_message: str, bot_reply: str, current_intimacy: int) -> int:
    prompt = f"""
你是「對話親密度影響」的裁判。請綜合使用者訊息與機器人回覆，評估這次互動對親密度的變化。
回傳 JSON，整數介於 -2 到 +2。

規則（越高越親密）：
- 明確讚美、撒嬌、示好、感謝、分享私事或脆弱 → +1 ~ +2
- 普通閒聊或資訊型問題 → 0
- 明確拒絕、批評、貶低、嘲諷 → -1 ~ -2
- 若雙方語氣一致偏甜/親暱，可適度提高；若機器人語氣冷淡、拒絕，則降低。
- 不要因為單一正向詞就極端加分；請考慮上下文完整語意。

輸入：
使用者：「{user_message}」
機器人回覆：「{bot_reply}」
目前親密度：{current_intimacy}

只回傳 JSON 結構如下:
{{
  "intimacy_change": -2
}}
"""
    data = await call_llm_and_parse_json(prompt)
    if data:
        change = int(data.get("intimacy_change", 0))
        return max(-2, min(2, change))
    return 0

async def update_memory(user_id: str, user_message: str):
    facts = await extract_facts_from_llm(user_message)
    if not facts:
        facts = extract_facts(user_message)
    for fact in facts:
        if len(fact.strip()) >= 4:
            await append_memory(user_id, fact)

async def update_intimacy(user_id: str, user_message: str, bot_reply: str) -> Tuple[int, int]:
    user_data = await get_user_data(user_id)
    current_intimacy = user_data["intimacy"]

    change_by_llm = await evaluate_intimacy_from_llm(user_message, bot_reply, current_intimacy)
    if change_by_llm == 0:
        change_by_llm = keyword_intimacy_fallback(user_message)

    emo = extract_emotion_tag(bot_reply)
    change_by_emotion = emotion_weight(emo)

    total_change = max(-2, min(2, change_by_llm + change_by_emotion))
    intimacy = await adjust_intimacy(user_id, total_change)
    return intimacy, total_change

async def generate_tts(bot_reply: str, llm_instance: Any) -> Optional[str]:
    try:
        from tts.voicevox import synthesize_with_translation 
        return await synthesize_with_translation(bot_reply, llm_instance)
    except Exception as e:
        logger.error(f"TTS 生成失敗：{e}")
        return None

# ==================== API 端點 ====================

@app.get("/")
async def index() -> HTMLResponse:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(base_dir, "templates", "index.html")
    return FileResponse(index_path)

@app.post("/chat")
async def chat(req: Request, payload: ChatPayload, background: BackgroundTasks):
    session = req.session
    user_id = ensure_user_id_in_session(session)
    user_message = (payload.message or "").strip()
    if not user_message:
        return JSONResponse({"error": "No message provided."}, status_code=400)

    # 0. 若開啟聯網搜尋，先執行 Tavily 檢索
    search_context = ""
    search_sources = []
    search_warning = None

    if payload.use_search:
        tavily_key = await get_user_tavily_key(user_id)
        if not tavily_key:
            search_warning = "尚未設定 Tavily API Key，已改用一般對話回覆（請至右上角「設定」填寫）"
        else:
            logger.info(f"[Chat] 使用者 {user_id} 啟用聯網搜尋: {user_message}")
            search_context, search_sources = await perform_chat_web_search(user_message, tavily_key)
            if not search_context:
                search_warning = "聯網搜尋未找到相關結果，已使用一般對話回覆"

# 0.5 端側檔案混合檢索 (Hybrid Local RAG: BM25 + BGE-Small)
    local_rag_context = ""
    local_rag_sources = []
    if payload.use_local_rag:
        from local_file_rag import LocalFileRAGManager
        rag_mgr = await _to_thread(LocalFileRAGManager.get_instance)
        local_rag_context, local_rag_sources = await _to_thread(
            rag_mgr.retrieve_hybrid,
            user_id,
            user_message,
            top_k=5,
            active_files=payload.active_files,
        )

    # 1. 生成 Prompt 與主要回覆
    system_prompt = await generate_system_prompt(user_id)

    context_blocks = []
    if local_rag_context:
        context_blocks.append(
            f"【端側知識庫檢索結果（精確行號 / 論文頁碼）】\n{local_rag_context}\n\n"
            "【重要檢索回覆指示】\n"
            "1. 請精準引用上述檢索到的【檔案名稱】、【行號 (Lines X-Y)】或【論文頁碼 (Page P)】與公式內容來進行 Code Review 或論文解析。\n"
            "2. 務必維持月讀醬微傲嬌又專業的女僕語氣（例如一邊吐槽主人的寫法或一邊得意地幫主人推導公式），且開頭必須加上 [emotion:tag]！"
        )
    if search_context:
        context_blocks.append(f"【即時網路搜尋結果】\n{search_context}")

    if context_blocks:
        joined_ctx = "\n\n".join(context_blocks)
        llm_user_input = f"{joined_ctx}\n\n【主人的問題】\n{user_message}"
    else:
        llm_user_input = user_message

    messages = build_chat_messages(session.get("chat_history", []), llm_user_input, system_prompt)
    bot_reply = await call_llm(messages)

    # 2. 更新 Session（僅儲存乾淨的 user_message，防止超過瀏覽器 4KB Cookie 上限）
    history = session.get("chat_history", [])
    history.append({"user": user_message, "bot": bot_reply})
    session["chat_history"] = history[-MAX_MEMORY:]

    # 3. 將耗時的記憶萃取丟給背景任務
    background.add_task(update_memory, user_id, user_message)

    # 4. 親密度更新
    intimacy, total_change = await update_intimacy(user_id, user_message, bot_reply)

    # 5. 語音生成
    audio_url = await generate_tts(bot_reply, chat_model)

    return JSONResponse(
        {
            "reply": bot_reply,
            "audio_url": audio_url,
            "intimacy": intimacy,
            "intimacy_level": get_intimacy_level_name(intimacy),
            "intimacy_change": total_change,
            "sources": search_sources,
            "search_warning": search_warning,
        }
    )

@app.get("/get_intimacy")
async def get_intimacy(req: Request):
    user_id = ensure_user_id_in_session(req.session)
    data = await get_user_data(user_id)
    return JSONResponse(
        {
            "intimacy": data["intimacy"],
            "intimacy_level": get_intimacy_level_name(data["intimacy"]),
        }
    )

@app.post("/update_intimacy")
async def update_intimacy_api(req: Request, payload: IntimacyUpdatePayload):
    user_id = ensure_user_id_in_session(req.session)
    intimacy = await adjust_intimacy(user_id, payload.amount)
    return JSONResponse(
        {"intimacy": intimacy, "intimacy_level": get_intimacy_level_name(intimacy)}
    )

@app.post("/clear_session")
async def clear_session(req: Request):
    req.session.pop("chat_history", None)
    return JSONResponse({"message": "已清除對話記憶（短期記憶）"})

@app.post("/clear_memory")
async def clear_memory(req: Request):
    req.session.pop("chat_history", None)
    user_id = req.session.get("user_id")
    if user_id:
        async with MEMORY_LOCK:
            def _delete():
                with get_db_connection() as conn:
                    c = conn.cursor()
                    c.execute("DELETE FROM user_memory WHERE user_id = ?", (user_id,))
            await _to_thread(_delete)
            logger.info(f"已刪除使用者 {user_id} 的長期記憶")
    return JSONResponse({"message": "已清除使用者的所有記憶（長期 + 短期）"})

@app.get("/api/files")
async def list_uploaded_files(req: Request):
    user_id = ensure_user_id_in_session(req.session)
    from local_file_rag import LocalFileRAGManager
    rag_mgr = await _to_thread(LocalFileRAGManager.get_instance)
    files = await _to_thread(rag_mgr.list_files, user_id)
    return JSONResponse({"files": files})


@app.post("/api/files/upload")
async def upload_files_for_rag(req: Request, files: List[UploadFile] = File(...)):
    user_id = ensure_user_id_in_session(req.session)
    from local_file_rag import LocalFileRAGManager
    rag_mgr = await _to_thread(LocalFileRAGManager.get_instance)

    results = []
    for uf in files:
        content = await uf.read()
        if len(content) > 30 * 1024 * 1024:  # 放寬單檔上限至 30MB (支援長篇 arXiv PDF)
            results.append({"filename": uf.filename, "success": False, "error": "超過 30MB 限制"})
            continue
        res = await _to_thread(rag_mgr.ingest_file, user_id, uf.filename, content)
        results.append(res)

    all_files = await _to_thread(rag_mgr.list_files, user_id)
    return JSONResponse({"results": results, "files": all_files})


@app.delete("/api/files/{filename:path}")
async def delete_uploaded_file(req: Request, filename: str):
    user_id = ensure_user_id_in_session(req.session)
    from local_file_rag import LocalFileRAGManager
    rag_mgr = await _to_thread(LocalFileRAGManager.get_instance)
    ok = await _to_thread(rag_mgr.delete_file, user_id, filename)
    all_files = await _to_thread(rag_mgr.list_files, user_id)
    return JSONResponse({"success": ok, "files": all_files})

@app.post("/research/start")
async def start_research(req: Request, payload: ResearchPayload, background: BackgroundTasks):
    try:
        topic = payload.topic.strip()
        if not topic:
            return JSONResponse({"success": False, "error": "請輸入研究主題"}, status_code=400)
        
        user_id = ensure_user_id_in_session(req.session)
        
        def _fetch_user_settings():
            with get_db_connection() as conn:
                c = conn.cursor()
                c.execute("SELECT tavily_api_key, smart_llm, small_llm FROM user_settings WHERE user_id = ?", (user_id,))
                return c.fetchone()
                
        row = await _to_thread(_fetch_user_settings)
        tavily_api_key = row["tavily_api_key"] if row and row["tavily_api_key"] else ""
        
        # 檢查本地向量資料庫
        vector_store_root = os.path.join(os.getcwd(), "vector_stores")
        has_local_db = False
        
        if os.path.exists(vector_store_root) and os.path.isdir(vector_store_root):
            safe_topic_prefix = re.sub(r'[\\/:*?"<>|]', "_", topic).strip()
            try:
                for folder_name in os.listdir(vector_store_root):
                    folder_path = os.path.join(vector_store_root, folder_name)
                    if os.path.isdir(folder_path) and folder_name.startswith(safe_topic_prefix):
                        has_local_db = True
                        logger.info(f"[Research] 檢測到本地對應主題的資料庫: {folder_name}")
                        break
            except Exception as e:
                logger.warning(f"[Research] 檢查本地資料庫時發生錯誤: {e}")

        if not tavily_api_key and not has_local_db:
            error_msg = f"未偵測到主題「{topic}」的本地知識庫，且未設定 Tavily API Key，無法進行新搜尋。"
            logger.warning(f"[Research] 用戶 {user_id} 請求被阻擋: {error_msg}")
            return JSONResponse({
                "success": False,
                "error": error_msg,
                "type": "missing_config"  
            }, status_code=200)
        
        smart_llm = row["smart_llm"] if row and row["smart_llm"] else CHAT_MODEL
        small_llm = row["small_llm"] if row and row["small_llm"] else CHAT_MODEL
        
        job_id = str(uuid.uuid4())
        logger.info(f"[Research] 啟動任務 {job_id}: {topic} (用戶: {user_id})")
        
        from researcher.service import init_progress, run_research
        
        init_progress(job_id, topic, total_items=1)
        background.add_task(run_research, job_id, topic, payload.max_results, user_id, smart_llm, small_llm)
        
        return JSONResponse({"job_id": job_id, "topic": topic})
        
    except Exception as e:
        logger.error(f"[Research] 啟動任務失敗: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/research/progress/{job_id}")
async def get_research_progress(job_id: str):
    try:
        from researcher.service import get_progress
        progress = get_progress(job_id)
        return JSONResponse(progress)
    except Exception as e:
        logger.error(f"[Research] 查詢進度失敗: {e}")
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

@app.post("/research")
async def research_api_legacy(req: Request, payload: ResearchPayload):
    async with RESEARCH_LOCK:
        try:
            topic = payload.topic.strip()
            if not topic:
                return JSONResponse({"success": False, "error": "請輸入研究主題"}, status_code=400)
            
            user_id = ensure_user_id_in_session(req.session)
            
            def _fetch_llm_settings():
                with get_db_connection() as conn:
                    c = conn.cursor()
                    c.execute("SELECT smart_llm, small_llm FROM user_settings WHERE user_id = ?", (user_id,))
                    return c.fetchone()
                    
            row = await _to_thread(_fetch_llm_settings)
            smart_llm = row["smart_llm"] if row and row["smart_llm"] else CHAT_MODEL
            small_llm = row["small_llm"] if row and row["small_llm"] else CHAT_MODEL
            
            logger.info(f"[Research] 開始研究（舊版 API): {topic}")
            
            from researcher.service import run_research
            job_id = str(uuid.uuid4())
            
            result = await _to_thread(run_research, job_id, topic, payload.max_results, user_id, smart_llm, small_llm)
            
            return JSONResponse(result)
        except Exception as e:
            logger.error(f"[Research] 研究失敗: {e}")
            import traceback
            traceback.print_exc()
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/api/settings")
async def get_settings(request: Request):
    user_id = ensure_user_id_in_session(request.session)
    
    def _get():
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute("""
                SELECT theme, font_size, tavily_api_key, smart_llm, small_llm
                FROM user_settings WHERE user_id = ?
            """, (user_id,))
            return c.fetchone()
            
    row = await _to_thread(_get)
    
    if row:
        return JSONResponse({
            "theme": row["theme"],
            "fontSize": row["font_size"],
            "tavilyApiKey": row["tavily_api_key"] or "",
            "smartLLM": row["smart_llm"],
            "smallLLM": row["small_llm"]
        })
    else:
        return JSONResponse({
            "theme": "dark",
            "fontSize": 16,
            "tavilyApiKey": "",
            "smartLLM": CHAT_MODEL, 
            "smallLLM": CHAT_MODEL  
        })

@app.post("/api/settings")
async def save_settings(request: Request, settings_data: SettingsUpdateRequest):
    user_id = ensure_user_id_in_session(request.session)
    settings = settings_data.settings
    
    def _save():
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute("""
                INSERT OR REPLACE INTO user_settings 
                (user_id, theme, font_size, tavily_api_key, smart_llm, small_llm, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                user_id,
                settings.theme,
                settings.fontSize,
                settings.tavilyApiKey,
                settings.smartLLM,
                settings.smallLLM
            ))
            
    await _to_thread(_save)
    logger.info(f"使用者 {user_id} 已更新設定")
    
    return JSONResponse({
        "success": True,
        "message": "設定已儲存"
    })

@app.post("/api/settings/test-api")
async def test_tavily_api(request: Request):
    try:
        data = await request.json()
        api_key = data.get("apiKey")
        
        if not api_key:
            return JSONResponse({
                "success": False,
                "message": "請提供 API Key"
            }, status_code=400)
        
        if len(api_key) < 20:
            return JSONResponse({
                "success": False,
                "message": "API Key 格式不正確"
            }, status_code=400)
        
        return JSONResponse({
            "success": True,
            "message": "API Key 格式正確"
        })
        
    except Exception as e:
        logger.error(f"測試 API 失敗: {e}")
        return JSONResponse({
            "success": False,
            "message": str(e)
        }, status_code=500)
    