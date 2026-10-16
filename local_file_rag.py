# -*- coding: utf-8 -*-
"""
端側混合檢索引擎 (Hybrid Local RAG for Code & Papers)
支援：
1. C/C++/Python 等原始碼「精確行號 (Line Number)」切塊與 AST/語法感知分割
2. PDF 論文「精確頁碼 (Page Number)」解析與切塊
3. Hybrid Search: BM25 (支援 C++ 識別符、英文論文術語與中文 jieba) + Vector Search (BGE-small)
"""

import os
import re
import shutil
import hashlib
import logging
import threading
from typing import List, Dict, Any, Tuple, Optional

import jieba
from pypdf import PdfReader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("local_file_rag")

UPLOAD_DIR = os.path.join(os.getcwd(), "user_uploads")
VECTOR_DIR = os.path.join(os.getcwd(), "user_file_vectors")

CODE_EXTENSIONS = {".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hxx", ".py", ".js", ".ts", ".java", ".rs", ".go"}
TEXT_EXTENSIONS = {".md", ".txt", ".tex", ".json", ".yaml", ".yml", ".cmake"}


def _hybrid_code_and_text_tokenizer(text: str) -> List[str]:
    """
    專為 C++ 程式碼、arXiv 英文論文與繁體中文設計的混合分詞器：
    保留變數名稱 (如 bellman_ford, max_depth)、LaTeX 符號與英文單字，並對中文使用 jieba
    """
    tokens = []
    # 抽取英文單字、C++ snake_case 變數、數字與常用數學名詞
    eng_and_code = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+", text.lower())
    tokens.extend(eng_and_code)
    # 抽取中文區段交由 jieba 分詞
    chinese_segments = re.findall(r"[\u4e00-\u9fff]+", text)
    for seg in chinese_segments:
        tokens.extend(jieba.cut(seg))
    return tokens if tokens else [text]


class LocalFileRAGManager:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, embed_model_name: str = "BAAI/bge-small-zh-v1.5"):
        logger.info(f"[LocalRAG] 載入端側輕量 Embedding 模型: {embed_model_name}")
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        os.makedirs(VECTOR_DIR, exist_ok=True)
        # 在 Apple Silicon 上可將 device 設為 'mps' 或 'cpu'
        self.embedder = HuggingFaceEmbeddings(
            model_name=embed_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    @classmethod
    def get_instance(cls) -> "LocalFileRAGManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _get_user_paths(self, user_id: str) -> Tuple[str, str]:
        safe_uid = re.sub(r"[^a-zA-Z0-9_-]", "_", user_id)
        u_upload = os.path.join(UPLOAD_DIR, safe_uid)
        u_vector = os.path.join(VECTOR_DIR, safe_uid)
        os.makedirs(u_upload, exist_ok=True)
        os.makedirs(u_vector, exist_ok=True)
        return u_upload, u_vector

    # ==================== 檔案解析與行號/頁碼切塊 ====================

    def _chunk_code_with_line_numbers(self, content: str, filename: str, lines_per_chunk: int = 45, overlap_lines: int = 10) -> List[Document]:
        """將 C++ / Python 程式碼按行號切塊，並在內容開頭標註精確行號區間"""
        lines = content.splitlines()
        total_lines = len(lines)
        docs = []

        if total_lines == 0:
            return docs

        step = max(1, lines_per_chunk - overlap_lines)
        for start_idx in range(0, total_lines, step):
            end_idx = min(total_lines, start_idx + lines_per_chunk)
            start_line = start_idx + 1
            end_line = end_idx

            # 為每一行加上行號前綴，讓 LLM 能一秒定位到第幾行
            numbered_lines = [
                f"{line_no:4d} | {lines[line_no - 1]}"
                for line_no in range(start_line, end_line + 1)
            ]
            chunk_body = "\n".join(numbered_lines)
            header = f"【檔案：{filename} | 行號：Lines {start_line}-{end_line}】\n"

            docs.append(
                Document(
                    page_content=header + chunk_body,
                    metadata={
                        "source": filename,
                        "location": f"Lines {start_line}-{end_line}",
                        "type": "code",
                    },
                )
            )
            if end_idx >= total_lines:
                break
        return docs

    def _chunk_pdf_with_pages(self, file_path: str, filename: str) -> List[Document]:
        """解析 PDF 論文，保留頁碼與 LaTeX 數學公式上下文"""
        docs = []
        try:
            reader = PdfReader(file_path)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=900,
                chunk_overlap=150,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            for page_idx, page in enumerate(reader.pages):
                page_num = page_idx + 1
                raw_text = page.extract_text() or ""
                if not raw_text.strip():
                    continue
                sub_chunks = splitter.split_text(raw_text)
                for sub_idx, chunk in enumerate(sub_chunks, 1):
                    header = f"【論文檔案：{filename} | 頁碼：Page {page_num} (Part {sub_idx})】\n"
                    docs.append(
                        Document(
                            page_content=header + chunk,
                            metadata={
                                "source": filename,
                                "location": f"Page {page_num}",
                                "type": "pdf",
                            },
                        )
                    )
        except Exception as e:
            logger.error(f"[LocalRAG] 解析 PDF 失敗 ({filename}): {e}")
        return docs

    def _chunk_plain_text(self, content: str, filename: str) -> List[Document]:
        """解析 Markdown / LaTeX (.tex) / 一般文字檔，保留行號區間"""
        return self._chunk_code_with_line_numbers(content, filename, lines_per_chunk=50, overlap_lines=10)

    # ==================== 索引建立與管理 ====================

    def ingest_file(self, user_id: str, filename: str, file_bytes: bytes) -> Dict[str, Any]:
        """儲存上傳的檔案並建立端側向量索引"""
        u_upload, u_vector = self._get_user_paths(user_id)
        # 清理路徑確保支援資料夾上傳時的相對檔名 (例如 src/avl_tree.cpp -> src__avl_tree.cpp)
        safe_name = re.sub(r'[\\/:*?"<>|]+', "__", filename).lstrip("_")
        file_path = os.path.join(u_upload, safe_name)

        with open(file_path, "wb") as f:
            f.write(file_bytes)

        ext = os.path.splitext(safe_name)[1].lower()
        docs: List[Document] = []

        if ext == ".pdf":
            docs = self._chunk_pdf_with_pages(file_path, filename)
        elif ext in CODE_EXTENSIONS or ext in TEXT_EXTENSIONS:
            text_content = file_bytes.decode("utf-8", errors="replace")
            if ext in CODE_EXTENSIONS:
                docs = self._chunk_code_with_line_numbers(text_content, filename)
            else:
                docs = self._chunk_plain_text(text_content, filename)
        else:
            # 嘗試當作純文字處理
            try:
                text_content = file_bytes.decode("utf-8")
                docs = self._chunk_code_with_line_numbers(text_content, filename)
            except UnicodeDecodeError:
                os.remove(file_path)
                return {"success": False, "error": f"不支援的二進位檔案格式: {filename}"}

        if not docs:
            return {"success": False, "error": f"無法從檔案提取有效內容: {filename}"}

        vectorstore = Chroma(persist_directory=u_vector, embedding_function=self.embedder)
        # 先刪除同名檔案的舊切塊（支援覆寫更新）
        try:
            existing = vectorstore.get(where={"source": filename})
            if existing and existing.get("ids"):
                vectorstore.delete(ids=existing["ids"])
        except Exception:
            pass

        ids = [
            hashlib.md5(f"{filename}_{i}_{d.page_content[:30]}".encode("utf-8")).hexdigest()
            + f"_{i}"
            for i, d in enumerate(docs)
        ]
        vectorstore.add_documents(documents=docs, ids=ids)
        logger.info(f"[LocalRAG] 已為使用者 {user_id} 索引 {filename} (共 {len(docs)} 個 Chunks)")

        return {
            "success": True,
            "filename": filename,
            "size": len(file_bytes),
            "chunks": len(docs),
        }

    def list_files(self, user_id: str) -> List[Dict[str, Any]]:
        """列出使用者目前已索引的檔案清單"""
        _, u_vector = self._get_user_paths(user_id)
        sqlite_path = os.path.join(u_vector, "chroma.sqlite3")
        if not os.path.exists(sqlite_path):
            return []

        try:
            vectorstore = Chroma(persist_directory=u_vector, embedding_function=self.embedder)
            metadatas = vectorstore.get().get("metadatas", [])
            file_stats: Dict[str, int] = {}
            for meta in metadatas:
                if meta and "source" in meta:
                    src = meta["source"]
                    file_stats[src] = file_stats.get(src, 0) + 1
            return [{"name": k, "chunks": v} for k, v in file_stats.items()]
        except Exception as e:
            logger.error(f"[LocalRAG] 讀取檔案列表失敗: {e}")
            return []

    def delete_file(self, user_id: str, filename: str) -> bool:
        """從向量庫與硬碟刪除指定檔案"""
        u_upload, u_vector = self._get_user_paths(user_id)
        safe_name = re.sub(r'[\\/:*?"<>|]+', "__", filename).lstrip("_")
        file_path = os.path.join(u_upload, safe_name)
        if os.path.exists(file_path):
            os.remove(file_path)

        try:
            vectorstore = Chroma(persist_directory=u_vector, embedding_function=self.embedder)
            existing = vectorstore.get(where={"source": filename})
            if existing and existing.get("ids"):
                vectorstore.delete(ids=existing["ids"])
            return True
        except Exception as e:
            logger.error(f"[LocalRAG] 刪除檔案失敗 ({filename}): {e}")
            return False

    # ==================== 端側混合檢索 (BM25 + Vector) ====================

    def retrieve_hybrid(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        active_files: Optional[List[str]] = None,
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        執行 Hybrid Search (BM25 權重 0.45 + BGE 稠密向量權重 0.55)
        同時精準命中 C++ 函數名/LaTeX 變數 (BM25) 與語意概念 (Vector)
        """
        _, u_vector = self._get_user_paths(user_id)
        sqlite_path = os.path.join(u_vector, "chroma.sqlite3")
        if not os.path.exists(sqlite_path):
            return "", []

        try:
            vectorstore = Chroma(persist_directory=u_vector, embedding_function=self.embedder)
            db_data = vectorstore.get()
            raw_docs = db_data.get("documents", [])
            raw_metas = db_data.get("metadatas", [])

            if not raw_docs:
                return "", []

            # 根據前端勾選的檔案過濾（若無指定則檢索全部已上傳檔案）
            doc_objects = []
            for text, meta in zip(raw_docs, raw_metas):
                meta = meta or {}
                if active_files and meta.get("source") not in active_files:
                    continue
                doc_objects.append(Document(page_content=text, metadata=meta))

            if not doc_objects:
                return "", []

            k = min(top_k, len(doc_objects))
            search_kwargs = {"k": k}
            if active_files:
                search_kwargs["filter"] = {"source": {"$in": active_files}}

            vector_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

            bm25_retriever = BM25Retriever.from_documents(
                doc_objects, preprocess_func=_hybrid_code_and_text_tokenizer
            )
            bm25_retriever.k = k

            ensemble = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.45, 0.55],
            )
            matched_docs = ensemble.invoke(query)[:top_k]

            context_blocks = []
            sources = []
            seen_locs = set()

            for idx, doc in enumerate(matched_docs, 1):
                src = doc.metadata.get("source", "未知檔案")
                loc = doc.metadata.get("location", "")
                context_blocks.append(doc.page_content)
                badge_key = f"{src} ({loc})"
                if badge_key not in seen_locs:
                    seen_locs.add(badge_key)
                    sources.append({"title": badge_key, "url": ""})

            return "\n\n" + ("\n" + "=" * 40 + "\n").join(context_blocks), sources

        except Exception as e:
            logger.error(f"[LocalRAG] 混合檢索失敗: {e}")
            return "", []