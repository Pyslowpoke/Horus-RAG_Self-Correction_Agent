"""Legacy imports retained as adapters; verification requires an actual generated answer."""
from src.agents.retrieval_agent import make_retrieval_agent
from src.agents.fact_check_agent import make_fact_check_agent


def make_retrieve_node(hyde_retriever, hybrid_retriever):
    return make_retrieval_agent(hybrid_retriever, hyde_retriever)


def make_fact_check_node(fact_checker):
    return make_fact_check_agent(fact_checker)
