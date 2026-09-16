"""Backward-compatible constructor using the same graph as the application."""
from src.graph.multi_agent_graph import build_multi_agent_rag_graph


def build_rag_graph(hyde_retriever, hybrid_retriever, fact_checker):
    llm = fact_checker.llm_client
    return build_multi_agent_rag_graph(llm, llm, hybrid_retriever, None, fact_checker, None,
        hyde_retriever=hyde_retriever, max_retries=fact_checker.max_retries)
