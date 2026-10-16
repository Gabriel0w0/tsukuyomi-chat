# -*- coding: utf-8 -*-
"""
研究報告生成主程式
版本：v3.1.0 (MLX 原生加速與執行緒安全強化版)
變更：
- 改由獨立模組 mlx_engine 匯入 MLXChatModel，消除循環依賴與雙重載入
- 透過 _CachedRetriever 避免階段 1 與階段 3 重複檢索向量資料庫
- 修正並行生成時的重複內容檢查時機與 HTML 巢狀標籤閉合邏輯
"""

import os
import re
import html
import threading
import concurrent.futures
from datetime import datetime
from typing import Any, List, Tuple, Optional, Callable, Dict

# 匯入改造後的 RAG 管道與本機 MLX 封裝
from rag_pipeline import RAGPipeline
from mlx_engine import MLXChatModel, get_shared_llm
import researcher.parsers as parsers

# ========================================
# 參數 
# ========================================

OUTLINE_TEMPERATURE = 0.15 
CONTENT_TEMPERATURE = 0.3   

CONTENT_LENGTH_NORMAL = 2048
CONTENT_LENGTH_NESTED = 1024

DUPLICATE_CHECK_LENGTH = 100       
DUPLICATE_THRESHOLD = 2            
DUPLICATE_RECENT_WINDOW = 5000     

MIN_CONTEXT_LENGTH = 30            
FORCE_GENERATION_THRESHOLD = 500   

MAX_RESULTS = 20

REMOVE_PATTERNS = [
    re.compile(r'本章將.*?介紹'),
    re.compile(r'將詳細介紹'),
    re.compile(r'本研究旨在'),
    re.compile(r'以期.*'),
    re.compile(r'包括.*'),
    re.compile(r'涉及.*'),
    re.compile(r'等$'),
]
# ========================================
# 相容性介面 (供 researcher/service.py 匯入使用)
# ========================================

def get_content_llm(model_name: Optional[str] = None, is_nested: bool = False) -> MLXChatModel:
    """
    相容舊版 service.py 的介面：回傳全域共用的 MLXChatModel 實例。
    注意：為了避免傳入舊版預設值（如 'ministral-3:3B'）導致 HuggingFace 找不到路徑，
    若偵測到包含 ':' 的 Ollama 格式名稱，將自動改用 config.py 的 CHAT_MODEL。
    """
    if model_name and ":" in model_name:
        model_name = None  # 交由 get_shared_llm 自動讀取 config.CHAT_MODEL
        
    max_tokens = CONTENT_LENGTH_NESTED if is_nested else CONTENT_LENGTH_NORMAL
    llm = get_shared_llm(
        model_name_or_path=model_name,
        temperature=CONTENT_TEMPERATURE,
        max_tokens=max_tokens
    )
    return llm


def get_outline_llm(model_name: Optional[str] = None) -> MLXChatModel:
    """相容舊版 service.py 的大綱生成模型取得介面"""
    if model_name and ":" in model_name:
        model_name = None
        
    llm = get_shared_llm(
        model_name_or_path=model_name,
        temperature=OUTLINE_TEMPERATURE,
        max_tokens=CONTENT_LENGTH_NORMAL
    )
    return llm

# ========================================
# 工具類別與函數區塊 
# ========================================

class _CachedRetriever:
    """輕量級檢索結果快取包裝器：避免階段 1 檢索完後，pipeline 內部又重複檢索一次"""
    def __init__(self, docs: list):
        self.docs = docs

    def invoke(self, query: str) -> list:
        return self.docs

    def get_relevant_documents(self, query: str) -> list:
        return self.docs


def enhance_search_query(subsection: str, main_topic: str) -> str:
    if ":" in subsection or "：" in subsection:
        title_part = re.split('[：:]', subsection)[0].strip()
    else:
        title_part = subsection.strip()
    
    for pattern in REMOVE_PATTERNS:
        title_part = pattern.sub('', title_part)
    
    remove_words = ["分析", "討論", "評估", "預測", "說明", "介紹", "概述", "研究", "探討", "闡述"]
    for word in remove_words:
        if title_part.endswith(word):
            title_part = title_part[:-len(word)]
    
    title_part = re.sub(r'\s+', ' ', title_part).strip()
    words = title_part.split()
    core_keywords = " ".join(words[:5]) if len(words) > 5 else title_part
    if len(core_keywords) > 20:
        core_keywords = core_keywords[:20]
    
    enhanced_query = f"{main_topic} {core_keywords}".strip()
    domain_keywords = {
        "技術": ["技術", "原理"],
        "發展": ["發展", "歷史"],
        "演進": ["演進", "過程"],
        "歷程": ["歷程", "階段"],
        "創新": ["創新", "突破"],
        "影響": ["影響", "作用"],
        "應用": ["應用", "使用"],
        "商業": ["商業", "市場"],
        "文化": ["文化", "現象"],
        "社會": ["社會", "效應"],
        "未來": ["未來", "趨勢"],
        "問題": ["問題", "挑戰"],
    }
    
    for keyword, additions in domain_keywords.items():
        if keyword in core_keywords:
            enhanced_query += f" {additions[0]}"
            break
    
    if len(enhanced_query) > 50:
        enhanced_query = enhanced_query[:50].strip()
    
    return enhanced_query


def is_duplicate_content(new_content: str, existing_content: str, 
                         check_length: int = DUPLICATE_CHECK_LENGTH,
                         threshold: int = DUPLICATE_THRESHOLD) -> bool:
    new_clean = new_content.strip()[:check_length]
    if len(new_clean) < 20:
        return False
    recent_content = existing_content[-DUPLICATE_RECENT_WINDOW:]
    count = recent_content.count(new_clean)
    if count >= threshold:
        print(f"[WARN] 檢測到重複內容（出現 {count} 次）: {new_clean[:30]}...")
        return True
    return False


def is_content_sufficient(content: str, min_length: int = MIN_CONTEXT_LENGTH) -> bool:
    if not content or len(content.strip()) < min_length:
        return False
    
    insufficient_keywords = [
        "資料不足，無法針對此主題提供詳細說明",
        "資料不足，無法提供內容",
        "資料中沒有提到",
        "無法取得內容"
    ]
    for keyword in insufficient_keywords:
        if content.strip() == keyword or content.strip() == f"[{keyword}]":
            return False
    return True


def safe_retriever_invoke(retriever: Any, query: str):
    try:
        return retriever.invoke(query)
    except Exception as e:
        print(f"[ERROR] 檢索失敗（query={query[:50]}...）: {e}")
        return []


def print_section_header(title: str):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def normalize_markdown_headings(text: str, expected_level: str = "") -> str:
    lines = text.splitlines()
    fixed = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue 
        fixed.append(line)
    return "\n".join(fixed).strip()


def save_report(topic: str, content: str, outline: list, 
                success_items: int, total_items: int) -> Optional[str]:
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # 過濾不合法的檔案路徑字元
        safe_topic = re.sub(r'[\\/:*?"<>|]', '_', topic).strip()
        output_filename = f"report_{safe_topic}_{timestamp}.txt"
        success_rate = (success_items / total_items * 100) if total_items > 0 else 0
        
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(f"主題：{topic}\n")
            f.write("=" * 60 + "\n")
            f.write(f"生成時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"成功率：{success_rate:.1f}% ({success_items}/{total_items})\n")
            f.write(f"總字數：{len(content)} 字元\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("【報告大綱】\n\n")
            for section in outline:
                f.write(f"{section['title']}\n")
                for sub in section['subpoints']:
                    if isinstance(sub, str):
                        f.write(f"  - {sub}\n")
                    elif isinstance(sub, dict):
                        f.write(f"  • {sub['title']}\n")
                        for sub_sub in sub['subpoints']:
                            f.write(f"    - {sub_sub}\n")
            
            f.write("\n" + "=" * 60 + "\n\n")
            f.write("【報告內容】\n\n")
            f.write(content)
        
        print(f"\n[INFO] ✅ 報告已成功儲存至 {output_filename}")
        print(f"[INFO] 總字數: {len(content)} 字元")
        print(f"[INFO] 成功率: {success_rate:.1f}%")
        return output_filename
    except Exception as e:
        print(f"[ERROR] ❌ 儲存檔案失敗: {e}")
        return None


def check_retriever_quality(retriever: Any, topic: str):
    print_section_header("檢查搜尋結果品質")
    test_queries = [topic, f"{topic} 技術", f"{topic} 歷史", f"{topic} 影響", f"{topic} 應用"]
    total_docs = 0
    total_length = 0
    
    for query in test_queries:
        try:
            docs = safe_retriever_invoke(retriever, query)
            query_length = sum(len(doc.page_content) for doc in docs)
            total_docs += len(docs)
            total_length += query_length
            print(f"\n測試查詢: {query}")
            print(f"  找到文檔數: {len(docs)}")
            if docs:
                preview = docs[0].page_content[:100].replace('\n', ' ')
                print(f"  第一筆摘要: {preview}...")
        except Exception as e:
            print(f"  ❌ 查詢失敗: {e}")
            
    print(f"\n統計資訊:")
    print(f"  總文檔數: {total_docs}")

# ========================================
# 內容生成函數 (MLX 本機推論版)
# ========================================

def generate_content_with_fallback(pipeline: RAGPipeline, retriever: Any, 
                                   question: str, topic: str, 
                                   llm: MLXChatModel,
                                   is_nested: bool = False,
                                   existing_content: str = "") -> Tuple[str, bool]:
    heading_level = "####" if is_nested else "###"
    
    try:
        # 階段 1: 多重檢索
        enhanced_query = enhance_search_query(question, topic)
        docs = safe_retriever_invoke(retriever, enhanced_query)
        
        if not docs:
            clean_question = re.split('[：:]', question)[0].strip() if (":" in question or "：" in question) else question.strip()
            docs = safe_retriever_invoke(retriever, clean_question)
        
        if not docs:
            docs = safe_retriever_invoke(retriever, topic)
        
        # 階段 2: 檢查檢索結果
        if not docs:
            return "[資料不足，無法提供內容]", False
        
        context_text = pipeline._docs_to_context(docs)
        if len(context_text) < MIN_CONTEXT_LENGTH:
            return "[資料不足，無法提供內容]", False
        
        # 將已檢索到的 docs 包裝起來，避免 pipeline 內部重複檢索且確保用到 enhanced_query 的結果
        cached_retriever = _CachedRetriever(docs)
        
        # 階段 3: 生成（MLXChatModel.invoke 內部已具備全域 _gpu_lock 保護）
        content_result = pipeline.run_content_generation(
            retriever=cached_retriever,
            question=question,
            llm_override=llm,
            heading_level=heading_level 
        )
        
        if not content_result:
            return "[生成失敗]", False

        lines = content_result.strip().split('\n')
        if lines:
            first_line = lines[0].strip()
            if first_line.startswith('#') or (len(first_line) < 50 and question in first_line):
                content_result = '\n'.join(lines[1:]).strip()
        
        # 階段 4: 充足性檢查
        if not is_content_sufficient(content_result):
            if len(context_text) > FORCE_GENERATION_THRESHOLD:
                try:
                    content_result = pipeline.run_content_generation(
                        retriever=cached_retriever,
                        question=question,
                        llm_override=llm,
                        use_strict_mode=False,
                        heading_level=heading_level 
                    )
                    if not is_content_sufficient(content_result):
                        return "[資料不足，無法提供詳細說明]", False
                except Exception:
                    return "[資料不足，無法提供詳細說明]", False
            else:
                return "[資料不足，無法提供詳細說明]", False
        
        # 階段 5: 初步重複檢查（若為單執行緒呼叫時可立即攔截）
        if existing_content and is_duplicate_content(content_result, existing_content):
            return "[內容重複，已省略]", False
        
        return content_result, True
        
    except Exception as e:
        print(f"[ERROR] 生成內容時發生錯誤: {e}")
        return f"[生成失敗: {e}]", False

# ========================================
# 並行內容生成函數
# ========================================

def _process_content_item(
    item_tuple: Tuple[RAGPipeline, Any, str, str, MLXChatModel, bool, str, Callable, threading.Lock], 
    processed_count: List[int], 
    total_items: int
) -> Tuple[str, bool, str]:
    pipeline, retriever, question, topic, shared_llm, is_nested, existing_content, progress_callback, progress_lock = item_tuple
    
    content_result, success = generate_content_with_fallback(
        pipeline=pipeline,
        retriever=retriever,
        question=question,
        topic=topic,
        llm=shared_llm,
        is_nested=is_nested,
        existing_content=existing_content
    )
    
    content_result = normalize_markdown_headings(content_result)
    
    with progress_lock: 
        processed_count[0] += 1
        current_count = processed_count[0]

    progress_callback(
        current_count, 
        "生成內容", 
        f"處理進度 {current_count}/{total_items} 個小節：{question[:30]}..."
    )
    
    return content_result, success, "" 


def generate_content_concurrently(
    pipeline: RAGPipeline, 
    retriever: Any, 
    outline: List[Dict], 
    topic: str,
    shared_llm: MLXChatModel,
    max_concurrency: int,
    progress_callback: Callable[[int, str, str], None],
    progress_lock: threading.Lock
) -> Dict[str, Any]:
    total_items = sum(
        len(section['subpoints']) +
        sum(len(sub.get('subpoints', [])) if isinstance(sub, dict) else 0 for sub in section['subpoints'])
        for section in outline
    )
    
    processed_count = [0]
    cumulative_content = "" 
    full_html_content = ""
    failed_items = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        for section_idx, section in enumerate(outline, 1):
            section_title = section.get('title', f'章節 {section_idx}')
            full_html_content += f'''
<div class="report-section">
    <h2 class="section-title">
        <span class="section-number">{section_idx}.</span> {html.escape(section_title)}
    </h2>
'''
            sub_futures = []
            for sub in section['subpoints']:
                if isinstance(sub, str):
                    question = sub
                    task_args = (pipeline, retriever, question, topic, shared_llm, False, cumulative_content, progress_callback, progress_lock)
                    future = executor.submit(_process_content_item, task_args, processed_count, total_items)
                    sub_futures.append({"type": "standard", "question": question, "future": future})
                elif isinstance(sub, dict) and "title" in sub and "subpoints" in sub:
                    for nested in sub["subpoints"]:
                        question = nested
                        task_args = (pipeline, retriever, question, topic, shared_llm, True, cumulative_content, progress_callback, progress_lock)
                        future = executor.submit(_process_content_item, task_args, processed_count, total_items)
                        sub_futures.append({"type": "nested", "question": question, "future": future, "group_title": sub["title"]})

            # 追蹤目前開啟中的巢狀群組標題，解決多個巢狀群組與一般小節交錯時的 HTML 閉合問題
            current_nested_group: Optional[str] = None

            for sub_future_item in sub_futures:
                question = sub_future_item["question"]
                future = sub_future_item["future"]
                item_type = sub_future_item["type"]

                try:
                    content_result, success, _ = future.result()

                    # 在收集結果時與最新的 cumulative_content 進行二次重複比對，確保同章節並行生成也能攔截重複
                    if success and is_duplicate_content(content_result, cumulative_content):
                        content_result = "[內容重複，已省略]"
                        success = False

                    if success:
                        cumulative_content += content_result + "\n"
                    else:
                        failed_items += 1

                    cleaned_content = html.escape(content_result).replace('\n', '<br>')
                    display_content = cleaned_content if success else html.escape(content_result)

                    if item_type == "standard":
                        # 若先前有開啟的 nested-section，先將其閉合
                        if current_nested_group is not None:
                            full_html_content += '</div>\n'
                            current_nested_group = None

                        full_html_content += f'''
<div class="subsection">
    <h3>{html.escape(question)}</h3>
    <div class="subsection-content"><p>{display_content}</p></div>
</div>
'''
                    elif item_type == "nested":
                        group_title = sub_future_item["group_title"]
                        # 若切換到新的巢狀群組，先關閉舊群組再開啟新群組
                        if current_nested_group != group_title:
                            if current_nested_group is not None:
                                full_html_content += '</div>\n'
                            full_html_content += f'<div class="nested-section"><h3 class="nested-title">{html.escape(group_title)}</h3>\n'
                            current_nested_group = group_title

                        full_html_content += f'''
<div class="nested-item">
    <h4>{html.escape(question)}</h4>
    <p>{display_content}</p>
</div>
'''
                except Exception as e:
                    print(f"[ERROR] 任務失敗: {e}")
                    failed_items += 1
                    error_content = '[生成服務失敗]'
                    if item_type == "standard":
                        if current_nested_group is not None:
                            full_html_content += '</div>\n'
                            current_nested_group = None
                        full_html_content += f'<div class="subsection"><h3>{html.escape(question)}</h3><div class="subsection-content"><p>{error_content}</p></div></div>'
                    elif item_type == "nested":
                        group_title = sub_future_item["group_title"]
                        if current_nested_group != group_title:
                            if current_nested_group is not None:
                                full_html_content += '</div>\n'
                            full_html_content += f'<div class="nested-section"><h3 class="nested-title">{html.escape(group_title)}</h3>\n'
                            current_nested_group = group_title
                        full_html_content += f'<div class="nested-item"><h4>{html.escape(question)}</h4><p>{error_content}</p></div>'

            # 章節結束前，若仍有未關閉的 nested-section 則補上閉合標籤
            if current_nested_group is not None:
                full_html_content += '</div>\n'
            full_html_content += '</div>\n'

    return {
        'html_content': full_html_content,
        'processed_items': processed_count[0],
        'failed_items': failed_items,
        'total_items': total_items,
    }

# ========================================
# 主程式測試
# ========================================

def main():
    topic = "初音未來"
    model_path = "mlx-community/Ministral-3B-Instruct-4bit"
    
    print_section_header("研究報告生成系統 v3.1 (MLX 本地測試)")
    print(f"主題: {topic}")
    
    # 透過 get_shared_llm 取得單一 MLX 模型常駐實例
    shared_llm = get_shared_llm(model_name_or_path=model_path, temperature=CONTENT_TEMPERATURE)
    
    pipeline = RAGPipeline(
        query=topic,
        llm_instance=shared_llm,
        max_results=MAX_RESULTS,
        use_existing_db=True
    )
    
    retriever = pipeline.execute()
    if retriever is None:
        print("[ERROR] ❌ 無法建立向量資料庫")
        return
    
    print_section_header("生成報告大綱")
    output = pipeline.generate_structured_outline(retriever, topic)['result']
    outline = parsers.extract(response_output=output)
    
    if not outline:
        print("[ERROR] ❌ 大綱生成失敗")
        return
        
    print("\n[INFO] ✅ 大綱生成成功，開始執行內容組裝...")
    
    progress_lock = threading.Lock()
    def dummy_progress(count: int, stage: str, msg: str):
        print(f"[{stage}] ({count}) {msg}")
        
    result = generate_content_concurrently(
        pipeline=pipeline,
        retriever=retriever,
        outline=outline,
        topic=topic,
        shared_llm=shared_llm,
        max_concurrency=4,
        progress_callback=dummy_progress,
        progress_lock=progress_lock,
    )
    
    success_items = result['total_items'] - result['failed_items']
    save_report(topic, result['html_content'], outline, success_items, result['total_items'])


if __name__ == "__main__":
    main()