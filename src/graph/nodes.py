from src.graph.state import RAGState
def make_retrieve_node(hyde_retriever, hybrid_retriever):
    """工厂函数：创建检索节点"""
    def retrieve_node(state: RAGState) -> dict:
        query = state["query"]
        docs = hyde_retriever.retrieve(query, top_k=5)
        return {"retrieved_docs": docs}
    return retrieve_node
def make_fact_check_node(fact_checker):
    """工厂函数：创建核查节点"""
    def fact_check_node(state: RAGState) -> dict:
        result = fact_checker.execute_pipeline(state["query"], state["retrieved_docs"])
        return {
            "answer": result["final_answer"],
            "verification_log": result["verification_log"],
            "retry_count": result["retry_count"]
        }
    return fact_check_node