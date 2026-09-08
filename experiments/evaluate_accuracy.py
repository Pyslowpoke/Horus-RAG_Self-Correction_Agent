"""
评估实验脚本

对比三种方案：纯向量检索 / 混合检索 / 混合检索+事实核查
指标：答案忠实度、幻觉率
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import jieba

from src.generators.llm_client import FaultTolerantLLM
from src.generators.prompt_templates import GENERATION_PROMPT
from src.retrievers.hybrid_retriever import HybridRetriever
from src.verifiers.fact_checker import FactChecker
from src.graph.rag_graph import build_rag_graph
from src.retrievers.hyde_retriever import HyDERetriever

from experiments.test_data import TEST_CASES

# 加载系统身份
SYSTEM_IDENTITY = ""
try:
    with open(PROJECT_ROOT / "knowledge_base" / "system_profile.txt", encoding="utf-8") as f:
        SYSTEM_IDENTITY = f.read()
except FileNotFoundError:
    SYSTEM_IDENTITY = "系统名称：Horus（荷鲁斯）\nSlogan：Insight, not imagination."


def init_components():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    db = Chroma(persist_directory=str(PROJECT_ROOT / "chroma_db"), embedding_function=embeddings)

    primary_config = {"api_key": os.getenv("DEEPSEEK_API_KEY"), "api_base": "https://api.deepseek.com/v1", "model": "deepseek-chat"}
    fallback_config = {"api_key": os.getenv("SILICONFLOW_API_KEY"), "api_base": "https://api.siliconflow.cn/v1", "model": "Qwen/Qwen2.5-7B-Instruct"}
    llm = FaultTolerantLLM(primary_config, fallback_config)

    all_docs_list = db.get()["documents"]
    all_docs = [Document(page_content=doc) for doc in all_docs_list]
    tokenized_docs = [jieba.lcut(doc) for doc in all_docs_list]
    bm25_index = BM25Okapi(tokenized_docs)

    hybrid_retriever = HybridRetriever(db, bm25_index, all_docs)
    fact_checker = FactChecker(llm_client=llm, retriever=None, system_identity=SYSTEM_IDENTITY)
    hyde_retriever = HyDERetriever(llm, db)

    return db, llm, hybrid_retriever, fact_checker, hyde_retriever


def evaluate_baseline_vector(db, llm, query):
    """基线：纯向量检索 + 普通生成"""
    docs = db.similarity_search(query, k=3)
    context = "\n".join([f"[{i+1}] {doc.page_content}" for i, doc in enumerate(docs)])
    messages = [{"role": "user", "content": GENERATION_PROMPT.format(system_identity=SYSTEM_IDENTITY, context=context, query=query)}]
    return llm.generate(messages), docs


def evaluate_hybrid(hybrid_retriever, llm, query):
    """改进A：混合检索 + 普通生成"""
    docs = hybrid_retriever.retrieve(query, top_k=3)
    context = "\n".join([f"[{i+1}] {doc.page_content}" for i, doc in enumerate(docs)])
    messages = [{"role": "user", "content": GENERATION_PROMPT.format(system_identity=SYSTEM_IDENTITY, context=context, query=query)}]
    return llm.generate(messages), docs


def evaluate_hybrid_fact_check(rag_graph, query):
    """改进B：混合检索 + 事实核查"""
    result = rag_graph.invoke({"query": query})
    return result["answer"], result.get("retrieved_docs", []), result.get("verification_log", [])


def calculate_metrics(answer, expected_claims):
    """计算评估指标（简化版：检查关键词是否出现）"""
    supported = sum(1 for claim in expected_claims if claim in answer)
    faithfulness = supported / len(expected_claims) if expected_claims else 0
    return {"faithfulness": faithfulness, "hallucination": 1 - faithfulness}


def run_evaluation():
    print("=" * 60)
    print("RAG-2.0 评估实验")
    print("=" * 60)

    db, llm, hybrid_retriever, fact_checker, hyde_retriever = init_components()
    rag_graph = build_rag_graph(hyde_retriever, hybrid_retriever, fact_checker)

    results = {
        "baseline": {"faithfulness": [], "hallucination": []},
        "hybrid": {"faithfulness": [], "hallucination": []},
        "hybrid_fact_check": {"faithfulness": [], "hallucination": []}
    }

    for i, test_case in enumerate(TEST_CASES):
        query = test_case["query"]
        expected = test_case["expected_claims"]
        print(f"\n--- 测试 {i+1}/{len(TEST_CASES)}: {query} ---")

        # 方案1：纯向量检索
        answer1, _ = evaluate_baseline_vector(db, llm, query)
        m1 = calculate_metrics(answer1, expected)
        results["baseline"]["faithfulness"].append(m1["faithfulness"])
        results["baseline"]["hallucination"].append(m1["hallucination"])
        print(f"  基线     - 忠实度: {m1['faithfulness']:.2f}, 幻觉率: {m1['hallucination']:.2f}")

        # 方案2：混合检索
        answer2, _ = evaluate_hybrid(hybrid_retriever, llm, query)
        m2 = calculate_metrics(answer2, expected)
        results["hybrid"]["faithfulness"].append(m2["faithfulness"])
        results["hybrid"]["hallucination"].append(m2["hallucination"])
        print(f"  混合检索 - 忠实度: {m2['faithfulness']:.2f}, 幻觉率: {m2['hallucination']:.2f}")

        # 方案3：混合检索 + 事实核查
        answer3, _, _ = evaluate_hybrid_fact_check(rag_graph, query)
        m3 = calculate_metrics(answer3, expected)
        results["hybrid_fact_check"]["faithfulness"].append(m3["faithfulness"])
        results["hybrid_fact_check"]["hallucination"].append(m3["hallucination"])
        print(f"  混合+核查 - 忠实度: {m3['faithfulness']:.2f}, 幻觉率: {m3['hallucination']:.2f}")

    # 总结
    print("\n" + "=" * 60)
    print("评估总结")
    print("=" * 60)
    for method in ["baseline", "hybrid", "hybrid_fact_check"]:
        avg_faith = sum(results[method]["faithfulness"]) / len(results[method]["faithfulness"])
        avg_hall = sum(results[method]["hallucination"]) / len(results[method]["hallucination"])
        print(f"\n{method}:")
        print(f"  平均忠实度: {avg_faith:.2f}")
        print(f"  平均幻觉率: {avg_hall:.2f}")


if __name__ == "__main__":
    run_evaluation()
