"""One bounded retrieval/generation/verification graph."""
import logging
from functools import lru_cache
from pathlib import Path
from langgraph.graph import StateGraph, START, END
from src.agents.interfaces import AgentState
from src.agents import (router_agent, make_retrieval_agent, make_web_search_agent,
                        make_generation_agent, make_fact_check_agent, make_memory_agent)
from src.documents import unique_docs
from src.runtime import timed
logger = logging.getLogger(__name__)


class NullHybridRetriever:
    def retrieve(self, query, top_k=5):
        return []


@lru_cache(maxsize=1)
def _load_system_identity():
    path = Path(__file__).resolve().parents[2] / 'knowledge_base' / 'system_profile.txt'
    return path.read_text(encoding='utf-8') if path.exists() else '系统名称：Horus（荷鲁斯）'


def build_multi_agent_rag_graph(heavy_llm, light_llm, hybrid_retriever, web_search_retriever,
                               fact_checker, memory_bank, hyde_retriever=None, system_identity=None,
                               max_retries=1, min_docs_threshold=1, verification_enabled=True,
                               context_max_chars=6000, top_k=5, streaming=False, web_min_lexical_score=0.3):
    graph = StateGraph(AgentState)
    retriever = hybrid_retriever or NullHybridRetriever()
    identity = system_identity if system_identity is not None else _load_system_identity()

    def merge_context(state):
        local = state.get('retrieved_docs', [])
        web = state.get('web_docs', [])
        # Keep local evidence first in hybrid mode; web is a supplement.
        docs = unique_docs(local + web)
        parts, selected, remaining = [], [], context_max_chars
        for doc in docs[:top_k]:
            source = doc.metadata.get('url') or doc.metadata.get('source', '未知')
            prefix = f'[{len(selected) + 1}] 来源: {source}\n'
            available = remaining - len(prefix) - 2
            if available < 50:
                break
            content = doc.page_content[:available]
            # Preserve the exact evidence text used in generation and verification.
            from langchain_core.documents import Document
            selected.append(Document(page_content=content, metadata=dict(doc.metadata)))
            part = prefix + content
            parts.append(part)
            remaining -= len(part) + 2
        return {'context': '\n\n'.join(parts) if parts else '未找到相关文档。', 'all_docs': selected}

    def rewrite(state):
        answer = fact_checker._rewrite(state['answer'], state['failed_claims'], state['context'])
        return {'answer': answer, 'retry_count': state.get('retry_count', 0) + 1}

    def memory(state):
        if state.get('search_mode') == 'self_aware':
            return {'memory_context': ''}
        return make_memory_agent(memory_bank)(state)

    def after_memory(state):
        if state.get('search_mode') == 'self_aware':
            return 'generation'
        return 'web_search' if state.get('search_mode') == 'web' else 'retrieval'

    def after_retrieval(state):
        if state.get('search_mode') == 'hybrid' and not state.get('retrieved_docs') and web_search_retriever:
            return 'web_search'
        return 'merge_context'

    def after_check(state):
        if state.get('failed_claims') and state.get('retry_count', 0) < max_retries:
            return 'rewrite'
        return END

    nodes = {'router': router_agent, 'memory': memory,
             'retrieval': make_retrieval_agent(retriever, hyde_retriever),
             'web_search': make_web_search_agent(web_search_retriever,
                 retriever if hasattr(retriever, 'rank_and_filter') else None, web_min_lexical_score),
             'merge_context': merge_context,
             'generation': make_generation_agent(heavy_llm, system_identity=identity, streaming=streaming),
             'fact_check': make_fact_check_agent(fact_checker), 'rewrite': rewrite}
    for name, function in nodes.items():
        def measured(state, function=function, name=name):
            with timed('node.' + name):
                result = function(state)
                if name == 'generation' and not verification_enabled:
                    result.setdefault('verification_status', 'disabled')
                return result
        graph.add_node(name, measured)
    graph.add_edge(START, 'router')
    graph.add_edge('router', 'memory')
    graph.add_conditional_edges('memory', after_memory,
                               {name: name for name in ('retrieval', 'web_search', 'generation')})
    graph.add_conditional_edges('retrieval', after_retrieval,
                               {name: name for name in ('web_search', 'merge_context')})
    graph.add_edge('web_search', 'merge_context')
    graph.add_edge('merge_context', 'generation')
    graph.add_edge('generation', 'fact_check' if verification_enabled else END)
    graph.add_conditional_edges('fact_check', after_check, {'rewrite': 'rewrite', END: END})
    graph.add_edge('rewrite', 'fact_check')
    return graph.compile()
