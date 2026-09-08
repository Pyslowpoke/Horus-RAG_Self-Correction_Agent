"""
Multi-Agent 共享状态定义

所有字段均为可选（total=False），节点按需读取和更新。
"""

from typing import TypedDict, List, Literal, Dict
from langchain_core.documents import Document


class AgentState(TypedDict, total=False):
    # 用户输入
    query: str
    search_mode: Literal["local", "web", "hybrid"]
    chat_history: List[Dict[str, str]]  # {"role": "user/assistant", "content": "..."}
    preferences: Dict[str, str]

    # 上下文感知
    enhanced_context: str  # 环境感知上下文（时间、用户身份、对话历史）
    current_year: int      # 当前年份

    # 查询优化
    optimized_query: str

    # 记忆检索
    memory_context: str

    # 文档检索
    retrieved_docs: List[Document]
    web_docs: List[Document]
    all_docs: List[Document]
    top_k: int
    retrieval_count: int
    relevance_score: float  # 检索结果相关度（0.0~1.0）

    # 上下文与生成
    context: str
    answer: str

    # 事实核查与自纠正
    verification_log: List[Dict[str, str]]  # [{claim, verdict, evidence}]
    failed_claims: List[Dict[str, str]]
    retry_count: int
    max_retries: int

    # 调试
    execution_trace: List[str]
