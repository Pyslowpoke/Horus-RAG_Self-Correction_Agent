"""
混合检索器 - 向量检索 + BM25 关键词检索 + Reranker 重排序

流程：
1. 向量检索（取 top_k * 4 个候选）
2. BM25 检索（取 top_k * 4 个候选）
3. RRF 融合 → 取 top_k * 2 个候选
4. Reranker 重排序 → 取最终 top_k
"""

import numpy as np
import logging
from typing import List
from langchain_core.documents import Document
import jieba

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self, chroma_db, bm25_index, all_docs, reranker_model=None):
        """
        参数:
            chroma_db: ChromaDB 实例
            bm25_index: BM25 索引
            all_docs: 所有文档列表
            reranker_model: Reranker 模型实例（可选）
        """
        self.chroma_db = chroma_db
        self.bm25_index = bm25_index
        self.all_docs = all_docs
        self.reranker_model = reranker_model

        if len(all_docs) != bm25_index.corpus_size:
            logger.warning(f"[HybridRetriever] all_docs({len(all_docs)}) != bm25_index({bm25_index.corpus_size})")

    def _get_doc_id(self, doc: Document) -> str:
        if "id" in doc.metadata:
            return doc.metadata["id"]
        source = doc.metadata.get("source", "unknown")
        idx = doc.metadata.get("chunk_index", 0)
        return f"{source}_{idx}"

    def retrieve(self, query: str, top_k: int = 5) -> list:
        """
        执行混合检索 + Reranker 重排序

        流程：
        1. 向量检索（取 candidate_k 个候选）
        2. BM25 检索（取 candidate_k 个候选）
        3. RRF 融合 → 取 rerank_k 个候选
        4. Reranker 重排序 → 取最终 top_k
        """
        candidate_k = top_k * 4  # 扩大召回池
        rerank_k = top_k * 2     # Reranker 候选数

        # ---- 1. 向量检索 ----
        vector_results = self.chroma_db.similarity_search(query, k=candidate_k)

        # ---- 2. BM25 检索 ----
        query_tokens = jieba.lcut(query)
        bm25_scores = self.bm25_index.get_scores(query_tokens)
        top_indices = np.argsort(bm25_scores)[::-1][:candidate_k]
        bm25_results = [self.all_docs[i] for i in top_indices]

        # ---- 3. RRF 融合 ----
        rrf_k = 60
        rrf_scores = {}

        for rank, doc in enumerate(vector_results):
            doc_id = self._get_doc_id(doc)
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {"score": 0.0, "doc": doc}
            rrf_scores[doc_id]["score"] += 1.0 / (rrf_k + rank + 1)

        for rank, doc in enumerate(bm25_results):
            doc_id = self._get_doc_id(doc)
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {"score": 0.0, "doc": doc}
            rrf_scores[doc_id]["score"] += 1.0 / (rrf_k + rank + 1)

        # 取 rerank_k 个候选，准备给 Reranker
        sorted_items = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
        candidate_docs = [item["doc"] for item in sorted_items[:rerank_k]]

        if not candidate_docs:
            return []

        # ---- 4. Reranker 重排序 ----
        if self.reranker_model is not None:
            reranked_docs = self._rerank(query, candidate_docs, top_k)
            return reranked_docs
        else:
            # 没有 Reranker，直接返回 RRF 结果
            return candidate_docs[:top_k]

    def _rerank(self, query: str, docs: List[Document], top_k: int) -> List[Document]:
        """用 Reranker 对文档重新排序"""
        if not docs:
            return []

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            # 构建 (query, doc) 对
            pairs = [[query, doc.page_content] for doc in docs]

            # Tokenize
            inputs = self.reranker_model["tokenizer"](
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            # 推理
            with torch.no_grad():
                outputs = self.reranker_model["model"](**inputs)
                scores = outputs.logits.squeeze().tolist()

            # 如果只有一个文档，scores 可能是 float
            if isinstance(scores, float):
                scores = [scores]

            # 按分数排序
            doc_score_pairs = list(zip(docs, scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

            logger.info(f"[Reranker] 重排序完成: {len(docs)} 个候选 → {top_k} 个结果")

            return [doc for doc, _ in doc_score_pairs[:top_k]]

        except Exception as e:
            logger.error(f"[Reranker] 重排序失败: {e}，返回原始结果")
            return docs[:top_k]
