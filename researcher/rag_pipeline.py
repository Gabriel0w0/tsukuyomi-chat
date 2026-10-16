# -*- coding: utf-8 -*-
import os
import re
import hashlib
import jieba
from filelock import FileLock
from deep_translator import GoogleTranslator

# 向量資料庫與檢索
from langchain_chroma import Chroma
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# 本機 Embedding（取代 OllamaEmbeddings，不需開 Ollama 服務）
from langchain_huggingface import HuggingFaceEmbeddings

# 提示詞模板
from researcher.prompts import PromptTemps


class RAGPipeline:
    """RAG (Retrieval-Augmented Generation) 管道
    版本: 3.0.0 (MLX 原生加速版)
    - 移除 Ollama 依賴，改用本機 MLXChatModel 與 HuggingFace Embedding
    - 保留 Hybrid Search (Vector + BM25 with jieba)
    - 支援結構化大綱與長文小節生成
    """
    
    def __init__(self, query: str, llm_instance,
                 embed_model_name: str = "BAAI/bge-small-zh-v1.5",
                 max_results: int = 5,
                 base_directory: str = "vector_stores",
                 use_existing_db: bool = False,
                 tavily_api_key: str = None,
                 enable_hybrid_search: bool = True):
        """
        初始化 RAG Pipeline
        :param llm_instance: 傳入在 app.py 初始化好的 MLXChatModel 實例，避免重複載入權重佔用記憶體
        """
        self.query = query
        self.llm = llm_instance  # 直接複用本機 MLXChatModel
        
        # 本地 Embedding：在 Apple Silicon (MPS/CPU) 上極速運行，完全無需 Ollama
        print(f"[INFO] 🧠 載入本機 Embedding 模型: {embed_model_name}")
        self.embedder = HuggingFaceEmbeddings(
            model_name=embed_model_name,
            model_kwargs={'device': 'cpu'}, # 也可設為 'mps'
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.max_results = max_results
        self.use_existing_db = use_existing_db
        self.enable_hybrid_search = enable_hybrid_search
        
        # 動態設定 Tavily API Key
        self.tavily_api_key = tavily_api_key
        if self.tavily_api_key:
            os.environ["TAVILY_API_KEY"] = self.tavily_api_key
            print(f"[INFO] 🔑 使用自訂 Tavily API Key")
        elif "TAVILY_API_KEY" not in os.environ:
            print(f"[WARN] ⚠️ 未設定 Tavily API Key，搜尋功能可能無法使用")
            
        # 根據 query 生成專屬資料庫路徑
        self.persist_directory = self._generate_db_path(query, base_directory)
        print(f"[INFO] 📁 向量資料庫路徑: {self.persist_directory}")
        
        self.search_targets = [
            ("tw", "zh-TW"), ("us", "en"), ("uk", "en"),
            ("ca", "en"), ("jp", "ja"), ("kr", "ko")
        ]

    def _generate_db_path(self, query: str, base_dir: str) -> str:
        """根據研究主題生成資料庫路徑 (支援模糊比對)"""
        clean_query = re.sub(r'[^\w\s\u4e00-\u9fff]', '', query, flags=re.UNICODE)
        clean_query = re.sub(r'\s+', '_', clean_query.strip())
        if len(clean_query) > 50:
            clean_query = clean_query[:50]
            
        query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()[:8]
        expected_db_name = f"{clean_query}_{query_hash}"
        full_path = os.path.join(base_dir, expected_db_name)

        if self.use_existing_db and not os.path.exists(full_path) and os.path.exists(base_dir):
            safe_prefix = clean_query + "_"
            candidates = [f for f in os.listdir(base_dir) if f.startswith(safe_prefix)]
            if candidates:
                fuzzy_path = os.path.join(base_dir, candidates[0])
                print(f"[INFO] 模糊比對成功！轉向使用現有資料庫: {candidates[0]}")
                return fuzzy_path
            
        return full_path

    def load_existing_vectorstore(self):  
        """載入現存的向量資料庫"""
        sqlite_path = os.path.join(self.persist_directory, "chroma.sqlite3")
        
        if not os.path.exists(sqlite_path):
            print(f"[WARN] 向量資料庫不存在: {self.persist_directory}")
            return None
        
        if os.path.getsize(sqlite_path) < 1024:
            print(f"[ERROR] 資料庫檔案損壞或為空")
            return None
        
        try:
            print(f"[INFO] 載入現存向量資料庫: {self.persist_directory}")
            vectorstore = Chroma(
                persist_directory=self.persist_directory, 
                embedding_function=self.embedder
            )
            
            existing_docs = vectorstore.get().get('documents', [])
            doc_count = len(existing_docs)
            if doc_count == 0:
                print(f"[WARN] 資料庫為空，建議重新搜尋")
                return None
            
            print(f"[INFO] ✅ 成功載入 {doc_count} 筆文檔")
            return vectorstore
        except Exception as e:
            print(f"[ERROR] 載入失敗: {e}")
            return None

    def retrieve_context(self):
        """使用 Tavily 搜尋多國資料"""
        print("[INFO] 使用 Tavily 搜尋多國資料中...")
        all_results = []
        lang_map = {"zh-TW": "zh", "en": "en", "ja": "ja", "ko": "ko"}
        
        for country, lang in self.search_targets:
            lang_code = lang_map.get(lang, "en")
            items = self.tavily_search(self.query, max_results=self.max_results, language=lang_code)
            all_results.extend(items)
        
        if not all_results:
            print("[WARN] 各國皆無結果，嘗試翻譯 query 為英文再搜尋...")
            try:
                query_en = GoogleTranslator(source='auto', target='en').translate(self.query)
            except Exception:
                query_en = self.query
            all_results = self.tavily_search(query_en, max_results=self.max_results, language="en")
        
        print(f"[INFO] 共獲得 {len(all_results)} 筆搜尋結果。")
        return "\n\n".join([str(item) for item in all_results])

    def tavily_search(self, query, max_results=5, language="en"):
        try:
            search = TavilySearch(max_results=max_results, language=language)
            result = search.run(query)
            if isinstance(result, dict) and "results" in result:
                return result["results"]
            elif isinstance(result, list):
                return result
            return [result] if result else []
        except Exception as e:
            print(f"[ERROR] Tavily 搜尋失敗: {e}")
            return []

    def split_chunks(self, text, chunk_size=500, chunk_overlap=50):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return splitter.split_text(text)

    def build_vectorstore(self, chunks):
        """建立或更新向量資料庫"""
        def hash_text(text):
            return hashlib.md5(text.encode('utf-8')).hexdigest()
        
        lock_path = os.path.join(self.persist_directory, "chroma.lock")
        os.makedirs(self.persist_directory, exist_ok=True)
        
        with FileLock(lock_path, timeout=10):
            sqlite_path = os.path.join(self.persist_directory, "chroma.sqlite3")
            if os.path.exists(sqlite_path):
                print("[INFO] 向量資料庫已存在，執行追加比對...")
                vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedder
                )
                try:
                    existing_docs = vectorstore.get().get('documents', [])
                    existing_hashes = set(hash_text(doc) for doc in existing_docs)
                    new_chunks = [c for c in chunks if hash_text(c) not in existing_hashes]
                    if new_chunks:
                        vectorstore.add_texts(new_chunks)
                except Exception as e:
                    print(f"[WARN] 增量寫入異常，重新建立: {e}")
                    vectorstore = Chroma.from_texts(
                        texts=chunks,
                        embedding=self.embedder,
                        persist_directory=self.persist_directory
                    )
            else:
                vectorstore = Chroma.from_texts(
                    texts=chunks,
                    embedding=self.embedder,
                    persist_directory=self.persist_directory
                )
        return vectorstore

    def _chinese_tokenizer(self, text: str):
        return list(jieba.cut(text))

    def _build_hybrid_retriever(self, vectorstore):
        """建立混合檢索器 (Vector + BM25)"""
        try:
            vector_retriever = vectorstore.as_retriever(search_kwargs={"k": self.max_results})
            all_docs_content = vectorstore.get().get('documents', [])
            
            if not all_docs_content:
                return vector_retriever
            
            doc_objects = [Document(page_content=txt) for txt in all_docs_content]
            bm25_retriever = BM25Retriever.from_documents(
                doc_objects,
                preprocess_func=self._chinese_tokenizer
            )
            bm25_retriever.k = self.max_results
            
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.4, 0.6]
            )
            return ensemble_retriever
        except Exception as e:
            print(f"[ERROR] 混合檢索器建立失敗，降級為純向量: {e}")
            return vectorstore.as_retriever(search_kwargs={"k": self.max_results})

    def _docs_to_context(self, docs):
        return "\n\n".join([doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in docs])

    # ==================== 核心呼叫封裝 (脫離 LCEL) ====================

    def _call_model(self, system_text: str, user_text: str, llm_override=None) -> str:
        """直接使用本機 MLXChatModel 進行推論"""
        active_llm = llm_override or self.llm
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text}
        ]
        return active_llm.invoke(messages)

    def run_qa(self, retriever, question: str):
        docs = retriever.invoke(question)
        context = self._docs_to_context(docs)
        user_prompt = f"根據以下上下文回答問題。如果無法從上下文判斷，請說明「資訊不足」。\n\n上下文:\n{context}\n\n問題: {question}\n回答:"
        return self._call_model("你是專業的研究助手。", user_prompt)

    def generate_structured_outline(self, retriever, topic: str):
        """生成結構化大綱"""
        try:
            prompt_obj = PromptTemps.get_langgraph_outline_prompt()
            template_str = prompt_obj.template
        except Exception:
            template_str = (
                "請為以下主題生成結構化大綱，使用 markdown 格式，包含主要章節和小節：\n"
                "背景資料:\n{context}\n\n主題: {question}\n\n請提供詳細的目錄結構："
            )
        
        docs = retriever.invoke(topic)
        context = self._docs_to_context(docs)
        user_prompt = template_str.replace("{context}", context).replace("{question}", topic)
        
        result = self._call_model("你是專業的內容規劃師。", user_prompt)
        return {"result": result}

    def run_content_generation(self, retriever, question: str, llm_override=None, use_strict_mode=False, heading_level="###"):
        """撰寫報告小節內容"""
        try:
            if use_strict_mode:
                prompt_obj = PromptTemps.get_content_prompt_strict()
            else:
                prompt_obj = PromptTemps.get_content_prompt()
            template_str = prompt_obj.template
        except Exception:
            template_str = (
                "請根據上下文撰寫 3 到 5 句完整段落。\n"
                "背景資料:\n{context}\n\n章節標題: {question}\n\n"
                "請直接輸出純段落內容："
            )
            
        docs = retriever.invoke(question)
        context = self._docs_to_context(docs)
        
        user_prompt = (
            template_str
            .replace("{context}", context)
            .replace("{question}", question)
            .replace("{heading_level}", heading_level)
        )
        return self._call_model("你是專業的報告撰寫助手。", user_prompt, llm_override=llm_override)

    def execute(self):
        """執行完整檢索管線"""
        vectorstore = None
        if self.use_existing_db:
            vectorstore = self.load_existing_vectorstore()
            if vectorstore is None:
                self.use_existing_db = False
        
        if vectorstore is None:
            context = self.retrieve_context()
            chunks = self.split_chunks(context)
            if not chunks:
                return None
            vectorstore = self.build_vectorstore(chunks)
        
        if self.enable_hybrid_search:
            return self._build_hybrid_retriever(vectorstore)
        return vectorstore.as_retriever(search_kwargs={"k": self.max_results})


if __name__ == "__main__":
    # 測試範例
    topics = [
        "初音未來",
        "AI & Machine Learning 2024",
        "台灣半導體產業的未來發展？"
    ]
    
    smart_llm = "ministral-3:3b"
    
    for topic in topics:
        print(f"\n{'='*60}")
        print(f"測試主題: {topic}")
        print(f"{'='*60}")
        
        pipeline = RAGPipeline(
            query=topic,
            model=smart_llm,
            temperature=0.15,
            num_predict=-1,
            enable_hybrid_search=True
        )
        
        # 查看生成的資料庫路徑
        print(f"資料庫路徑: {pipeline.persist_directory}")