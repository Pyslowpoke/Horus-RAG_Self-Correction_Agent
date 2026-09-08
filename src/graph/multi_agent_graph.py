"""
Multi-Agent RAG Graph

流程: START → router → memory
    → retrieval → merge_context → generation
    → fact_check → [rewrite → fact_check (循环)] 或 END

支持双模型策略：轻量模型(查询优化) + 重量模型(生成/核查)
"""

import os
import logging
from functools import lru_cache
from langgraph.graph import StateGraph, START, END
from src.agents.interfaces import AgentState

logger = logging.getLogger(__name__)
from src.agents import (
    router_agent,
    make_retrieval_agent,
    make_web_search_agent,
    make_generation_agent,
    make_fact_check_agent,
    make_memory_agent,
)


class NullHybridRetriever:
    """空混合检索器（web 模式占位）"""
    def retrieve(self, query, top_k=5):
        return []

class NullMemoryBank:
    """空记忆库（web 模式占位）"""
    def get_memory_context_prompt(self, query, top_k=3):
        return ""

def _wrap_if_none(obj, null_class):
    return obj if obj is not None else null_class()


@lru_cache(maxsize=1)
def _load_system_identity():
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(project_root, "knowledge_base", "system_profile.txt"), encoding="utf-8") as f:
            return f.read()
    except (FileNotFoundError, IOError):
        return "系统名称：Horus（荷鲁斯）\nSlogan：Insight, not imagination.\n核心能力：混合检索、HyDE、事实核查、自我纠正、黄金记忆库"


def build_multi_agent_rag_graph(
    heavy_llm,
    light_llm,
    hybrid_retriever,
    web_search_retriever,
    fact_checker,
    memory_bank,
    hyde_retriever=None,
    system_identity=None,
    max_retries=2,
    min_docs_threshold=3,
):
    hybrid_retriever = _wrap_if_none(hybrid_retriever, NullHybridRetriever)
    memory_bank = _wrap_if_none(memory_bank, NullMemoryBank)
    if system_identity is None:
        system_identity = _load_system_identity()

    graph = StateGraph(AgentState)

    # 创建 Agent 节点
    retrieval_agent = make_retrieval_agent(hybrid_retriever, hyde_retriever=hyde_retriever)
    web_search_agent = make_web_search_agent(web_search_retriever)
    generation_agent = make_generation_agent(heavy_llm, system_identity=system_identity)
    fact_check_agent = make_fact_check_agent(fact_checker)
    memory_agent = make_memory_agent(memory_bank)

    # 上下文合并节点
    def merge_context_node(state):
        local_docs = state.get("retrieved_docs", [])
        web_docs = state.get("web_docs", [])
        search_mode = state.get("search_mode", "local")

        # 混合模式：优先用联网结果，本地结果作为补充
        if search_mode == "hybrid" and web_docs:
            # 如果联网有结果，以联网为主
            primary_docs = web_docs
            supplement_docs = local_docs
        else:
            primary_docs = local_docs
            supplement_docs = web_docs

        all_docs = primary_docs + supplement_docs

        if not all_docs:
            return {"context": "未找到相关文档。", "all_docs": []}

        parts = []
        for i, d in enumerate(all_docs):
            source = d.metadata.get("source", "未知")
            tag = "🌐联网" if "web_search" in str(source) else "📚本地"
            parts.append(f"[{i+1}] [{tag}] {d.page_content[:500]}\n来源: {source}")

        return {"context": "\n\n".join(parts), "all_docs": all_docs}

    # 重写节点
    def rewrite_node(state):
        import json
        failed = state.get("failed_claims", [])
        if not failed:
            return {}
        failed_json = json.dumps(failed, ensure_ascii=False, indent=2)
        prompt = f"原始回答：{state.get('answer','')}\n\n以下陈述有问题：\n{failed_json}\n\n上下文：{state.get('context','')[:2000]}\n用户问题：{state.get('query','')}"
        new_answer = heavy_llm.generate([{"role": "user", "content": prompt}])
        return {"answer": new_answer, "retry_count": state.get("retry_count", 0) + 1, "failed_claims": []}

    # 条件边
    def route_after_memory(state):
        mode = state.get("search_mode", "local")
        if mode == "self_aware":
            return "generation"  # 自我认知类问题直接跳到生成
        if mode == "web":
            return "web_search"
        return "retrieval"

    def route_after_retrieval(state):
        mode = state.get("search_mode", "local")
        relevance = state.get("relevance_score", 0.0)
        docs = state.get("retrieved_docs", [])

        # 混合模式：相关度低或结果少时联网补充
        if mode == "hybrid":
            if relevance < 0.3 or len(docs) < min_docs_threshold:
                logger.info(f"[Router] 混合模式：相关度={relevance:.2f}, 文档数={len(docs)}，触发联网")
                return "web_search"
            return "merge_context"

        # 本地模式：直接合并
        return "merge_context"

    def route_after_fact_check(state):
        if state.get("failed_claims") and state.get("retry_count", 0) < max_retries:
            return "rewrite"
        return END

    # 添加节点
    graph.add_node("router", router_agent)
    graph.add_node("memory", memory_agent)
    graph.add_node("retrieval", retrieval_agent)
    graph.add_node("web_search", web_search_agent)
    graph.add_node("merge_context", merge_context_node)
    graph.add_node("generation", generation_agent)
    graph.add_node("fact_check", fact_check_agent)
    graph.add_node("rewrite", rewrite_node)

    # 添加边
    graph.add_edge(START, "router")
    graph.add_edge("router", "memory")

    graph.add_conditional_edges("memory", route_after_memory, {"retrieval": "retrieval", "web_search": "web_search"})
    graph.add_conditional_edges("retrieval", route_after_retrieval, {"web_search": "web_search", "merge_context": "merge_context"})

    graph.add_edge("web_search", "merge_context")
    graph.add_edge("merge_context", "generation")
    graph.add_edge("generation", "fact_check")
    graph.add_conditional_edges("fact_check", route_after_fact_check, {"rewrite": "rewrite", END: END})
    graph.add_edge("rewrite", "fact_check")

    return graph.compile()
