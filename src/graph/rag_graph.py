from langgraph.graph import StateGraph, START, END
from src.graph.state import RAGState
from src.graph.nodes import make_retrieve_node, make_fact_check_node
def build_rag_graph(hyde_retriever, hybrid_retriever, fact_checker):
    """构建 RAG 流水线图"""
    # 1. 创建图
    graph = StateGraph(RAGState)
    # 2. 创建节点（用工厂函数注入依赖）
    retrieve_node = make_retrieve_node(hyde_retriever, hybrid_retriever)
    fact_check_node = make_fact_check_node(fact_checker)
    # 3. 添加节点
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("fact_check", fact_check_node)
    # 4. 添加连接
    graph.add_edge(START, "retrieve")       # 起点 → 检索
    graph.add_edge("retrieve", "fact_check") # 检索 → 核查
    graph.add_edge("fact_check", END)        # 核查 → 终点
    # 5. 编译图
    return graph.compile()