"""
Web Search Agent

职责：执行互联网搜索
"""

from src.agents.interfaces import AgentState


def make_web_search_agent(web_search_retriever):
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
        docs = web_search_retriever.retrieve(query, max_results=5)
        
        return {"web_docs": docs}
    
    return web_search_agent
