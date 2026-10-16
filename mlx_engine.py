# -*- coding: utf-8 -*-
"""
MLX 本機模型推論引擎 (Singleton & Thread-Safe)
負責統一管理 Apple Silicon Metal GPU 推論鎖與常駐模型實例
"""

import logging
import threading
from typing import List, Dict, Optional
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

logger = logging.getLogger("mlx_engine")


class MLXChatModel:
    # 類別層級的全域 Metal GPU 推論鎖，防止多執行緒同時呼叫 Metal 導致崩潰
    _gpu_lock = threading.Lock()

    def __init__(
        self,
        model_name_or_path: str,
        temperature: float = 0.35,
        top_p: float = 0.85,
        max_tokens: int = 2048,
    ):
        logger.info(f"正在載入 MLX 模型至 Apple Silicon: {model_name_or_path}")
        self.model_name_or_path = model_name_or_path
        self.model, self.tokenizer = load(model_name_or_path)
        self.default_temperature = temperature
        self.default_top_p = top_p
        self.max_tokens = max_tokens
        logger.info("MLX 模型載入完成！")

    def invoke(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        temp = self.default_temperature if temperature is None else temperature
        tp = self.default_top_p if top_p is None else top_p
        sampler = make_sampler(temp=temp, top_p=tp)

        # 強制排隊進入 Metal GPU 推論
        with self._gpu_lock:
            response_text = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                sampler=sampler,
                max_tokens=max_tokens or self.max_tokens,
                verbose=False,
            )

        print("=== 實際送給模型的 Prompt ===")
        print(prompt)
        print("============================")

        return response_text.strip()


# 全域單例實例快取
_shared_llm_instance: Optional[MLXChatModel] = None
_instance_lock = threading.Lock()


def get_shared_llm(
    model_name_or_path: Optional[str] = None,
    temperature: float = 0.35,
    top_p: float = 0.85,
    max_tokens: int = 2048,
) -> MLXChatModel:
    """取得或初始化全域共用的 MLXChatModel 實例，避免重複載入塞爆記憶體"""
    global _shared_llm_instance
    with _instance_lock:
        if _shared_llm_instance is None:
            if not model_name_or_path:
                from config import CHAT_MODEL
                model_name_or_path = CHAT_MODEL
            _shared_llm_instance = MLXChatModel(
                model_name_or_path=model_name_or_path,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        return _shared_llm_instance
    