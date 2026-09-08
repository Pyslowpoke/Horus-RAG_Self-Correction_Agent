"""
Fact Check Agent

职责：调用事实核查器，返回核查结果
"""

import logging
from typing import Dict, Any

from src.agents.interfaces import AgentState

logger = logging.getLogger(__name__)

# 无需核查的回答模式
SKIP_CHECK_PATTERNS = [
    "无法回答",
    "没有相关信息",
    "未找到",
    "暂时不可用",
    "出现异常",
    "请稍后重试",
]


def make_fact_check_agent(fact_checker):
    def fact_check_agent(state: AgentState) -> Dict[str, Any]:
        query = state.get("query", "")
        initial_answer = state.get("answer", "")
        context = state.get("context", "")
        search_mode = state.get("search_mode", "local")

        # 自我认知类问题跳过核查
        if search_mode == "self_aware":
            return {
                "answer": initial_answer,
                "verification_log": [],
                "failed_claims": [],
                "retry_count": 0,
            }

        # 无回答则跳过
        if not initial_answer:
            return {"verification_log": [], "failed_claims": [], "retry_count": 0}

        # "无法回答"类回答跳过核查
        if any(pattern in initial_answer for pattern in SKIP_CHECK_PATTERNS):
            logger.info("[FactCheckAgent] 跳过核查（无法回答类）")
            return {
                "answer": initial_answer,
                "verification_log": [],
                "failed_claims": [],
                "retry_count": 0,
            }

        # 无参考资料则跳过
        if not context or context == "未找到相关文档。":
            return {
                "answer": initial_answer,
                "verification_log": [],
                "failed_claims": [],
                "retry_count": 0,
            }

        # 执行核查
        try:
            result = fact_checker.execute_pipeline(query, initial_answer, context)
            failed = [c for c in result["verification_log"] if c.get("verdict") in ("矛盾", "证据不足")]
            return {
                "answer": result["final_answer"],
                "verification_log": result["verification_log"],
                "failed_claims": failed,
                "retry_count": result["retry_count"],
            }
        except Exception as e:
            logger.error("[FactCheckAgent] 核查失败: %s", str(e))
            return {
                "answer": initial_answer,
                "verification_log": [],
                "failed_claims": [],
                "retry_count": 0,
            }

    return fact_check_agent
