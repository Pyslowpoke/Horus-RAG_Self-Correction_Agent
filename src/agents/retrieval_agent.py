"""Retrieve original queries first; use HyDE only when evidence is insufficient."""
import logging
from src.documents import lexical_relevance
from src.runtime import RequestTimeout
logger = logging.getLogger(__name__)


def _compute_relevance(query, docs):
    return max((float(d.metadata.get('relevance_score', lexical_relevance(query, d)))
                for d in docs), default=0.0)


def make_retrieval_agent(hybrid_retriever, hyde_retriever=None, min_results=1, short_query_threshold=5):
    def retrieval_agent(state):
        query = state.get('optimized_query') or state.get('query', '')
        top_k = state.get('top_k', 5)
        try:
            docs = hybrid_retriever.retrieve(query, top_k=top_k) or []
            if not docs and hyde_retriever is not None:
                extra = hyde_retriever.retrieve(query, top_k=top_k)
                # Re-score against the actual question, not the hypothetical answer.
                docs = hybrid_retriever.retrieve(query, top_k=top_k, extra_docs=extra)
            return {'retrieved_docs': docs, 'retrieval_count': len(docs),
                    'relevance_score': _compute_relevance(query, docs)}
        except RequestTimeout:
            raise
        except Exception:
            logger.exception('检索失败')
            return {'retrieved_docs': [], 'retrieval_count': 0, 'relevance_score': 0.0,
                    'retrieval_error': '本地检索失败'}
    return retrieval_agent
