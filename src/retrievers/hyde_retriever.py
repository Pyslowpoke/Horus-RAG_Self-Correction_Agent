"""
HyDE 检索器（Hypothetical Document Embeddings）

流程：用户问题 → LLM 生成假设性答案 → 用假设答案做向量检索
"""

from typing import List
from langchain_core.documents import Document
from src.generators.prompt_templates import HYDE_PROMPT
from src.generators.llm_client import FaultTolerantLLM


class HyDERetriever:
    def __init__(self, llm_client: FaultTolerantLLM, vector_store,
                 min_hyde_length: int = 10, max_hyde_length: int = 200, max_tokens: int = 128):
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.min_hyde_length = min_hyde_length
        self.max_hyde_length = max_hyde_length
        self.max_tokens = max_tokens

    def _truncate_hyde_doc(self, text: str) -> str:
        """按句号截断到最大长度，保证语义完整"""
        if len(text) <= self.max_hyde_length:
            return text
        sentences = text.replace("。", "。\n").split("\n")
        truncated = ""
        for sent in sentences:
            if len(truncated) + len(sent) <= self.max_hyde_length:
                truncated += sent
            else:
                break
        return truncated if truncated else text[:self.max_hyde_length]

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """执行 HyDE 检索"""
        # 生成假设性答案
        messages = [{"role": "user", "content": HYDE_PROMPT.format(query=query)}]
        hyde_doc = self.llm_client.generate(messages, max_tokens=self.max_tokens)

        # 太短则 fallback 到原始 query
        if not hyde_doc or len(hyde_doc.strip()) < self.min_hyde_length:
            return self.vector_store.similarity_search(query, k=top_k)

        # 截断 + 向量检索
        hyde_doc = self._truncate_hyde_doc(hyde_doc)
        return self.vector_store.similarity_search(hyde_doc, k=top_k)
