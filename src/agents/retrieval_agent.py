"""
Retrieval Agent

职责：根据查询选择检索策略，按需触发 HyDE
     返回检索结果 + 相关度分数，供路由决策
"""

import logging
import time
from typing import Dict, Any

from src.agents.interfaces import AgentState

logger = logging.getLogger(__name__)


def _compute_relevance(query: str, docs: list) -> float:
    """
    简单相关度计算：查询关键词在文档中的命中率

    返回 0.0 ~ 1.0，越高越相关
    """
    if not docs:
        return 0.0

    # 提取查询关键词（去掉停用词）
    stop_words = {"的", "了", "是", "在", "和", "有", "这", "那", "我", "你", "他", "她", "它",
                  "什么", "怎么", "如何", "为什么", "可以", "能", "会", "吗", "呢", "吧",
                  "a", "the", "is", "are", "was", "were", "what", "how", "why", "can", "do"}
    keywords = [w for w in query if w not in stop_words and len(w) > 1]

    if not keywords:
        return 1.0  # 没有关键词，默认相关

    # 统计关键词在文档中的命中
    all_text = " ".join([doc.page_content for doc in docs])
    hits = sum(1 for kw in keywords if kw in all_text)
    return hits / len(keywords)


def make_retrieval_agent(hybrid_retriever, hyde_retriever=None, min_results=3, short_query_threshold=5):
    """
    创建检索代理

    参数:
        hybrid_retriever: 混合检索器实例（必须）
        hyde_retriever: HyDE 检索器实例（可选，不传则禁用 HyDE）
        min_results: hybrid 检索结果少于该值时触发 HyDE
        short_query_threshold: 查询长度小于该值时直接走 HyDE

    返回:
        retrieval_agent 函数
    """
    def retrieval_agent(state: AgentState) -> Dict[str, Any]:
        query = state.get("optimized_query") or state.get("query", "")
        top_k = state.get("top_k", 5)

        if hybrid_retriever is None:
            logger.warning("[RetrievalAgent] hybrid_retriever 为 None")
            return {"retrieved_docs": [], "retrieval_count": 0, "relevance_score": 0.0}

        try:
            t0 = time.time()

            # 短查询直接走 HyDE
            if hyde_retriever is not None and len(query) < short_query_threshold:
                logger.info("[RetrievalAgent] 短查询，直接走 HyDE: '%s'", query)
                docs = hyde_retriever.retrieve(query, top_k=top_k)
                relevance = _compute_relevance(query, docs)
                logger.info("[RetrievalAgent] HyDE 检索完成: %s 条, 相关度=%.2f", len(docs), relevance)
                return {"retrieved_docs": docs, "retrieval_count": len(docs), "relevance_score": relevance}

            # 长查询：先走 hybrid
            docs = hybrid_retriever.retrieve(query, top_k=top_k)
            if docs is None:
                docs = []

            relevance = _compute_relevance(query, docs)
            logger.info("[RetrievalAgent] Hybrid 检索: %s 条, 相关度=%.2f", len(docs), relevance)

            # 结果不足或相关度低时，走 HyDE 兜底
            if hyde_retriever is not None and (len(docs) < min_results or relevance < 0.3):
                logger.info("[RetrievalAgent] 触发 HyDE 兜底（数量=%s, 相关度=%.2f）", len(docs), relevance)
                hyde_docs = hyde_retriever.retrieve(query, top_k=top_k)
                if hyde_docs:
                    seen = set()
                    merged = []
                    for doc in docs + hyde_docs:
                        key = doc.page_content[:100]
                        if key not in seen:
                            seen.add(key)
                            merged.append(doc)
                    docs = merged
                    relevance = _compute_relevance(query, docs)

            logger.info("[RetrievalAgent] 最终结果: %s 条, 相关度=%.2f, 耗时=%.1fs", len(docs), relevance, time.time() - t0)
            return {"retrieved_docs": docs, "retrieval_count": len(docs), "relevance_score": relevance}

        except Exception as e:
            logger.error("[RetrievalAgent] 检索失败: %s", str(e), exc_info=True)
            return {"retrieved_docs": [], "retrieval_count": 0, "relevance_score": 0.0}

    return retrieval_agent
