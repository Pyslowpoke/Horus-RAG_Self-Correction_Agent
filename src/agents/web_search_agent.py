"""
Web Search Agent

职责：执行互联网搜索
"""

from src.agents.interfaces import AgentState
from src.runtime import RequestTimeout
import logging


def make_web_search_agent(web_search_retriever, evidence_ranker=None, min_lexical_score=0.3):
    """
    创建互联网搜索代理

    参数:
        web_search_retriever: 互联网搜索检索器实例

    返回:
        web_search_agent 函数
    """
    def web_search_agent(state: AgentState) -> dict:
        """
        互联网搜索代理：执行互联网搜索

        返回:
            dict: 包含 web_docs 的更新
        """
        # optimized_query 可能是空字符串，需要用 or 兜底
        query = state.get("optimized_query") or state.get("query", "")

        # 执行互联网搜索
        if web_search_retriever is None:
            return {"web_docs": [], "web_error": "联网搜索未配置"}
        try:
            docs = web_search_retriever.retrieve(query)
        except RequestTimeout:
            raise
        except Exception:
            logging.getLogger(__name__).exception('联网检索失败')
            return {"web_docs": [], "web_error": "联网搜索失败"}

        # Hybrid mode reuses its loaded reranker; web-only mode stays model-free.
        if evidence_ranker is not None:
            docs = evidence_ranker.rank_and_filter(query, docs, state.get('top_k', 5))
        else:
            from src.documents import lexical_relevance, unique_docs
            accepted = []
            for doc in unique_docs(docs):
                score = lexical_relevance(query, doc)
                doc.metadata.update(relevance_score=score, relevance_method='lexical_fallback')
                if score >= min_lexical_score and score > 0:
                    accepted.append(doc)
            docs = accepted[:state.get('top_k', 5)]

        return {"web_docs": docs}

    return web_search_agent
