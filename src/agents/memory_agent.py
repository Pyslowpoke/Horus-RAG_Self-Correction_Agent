"""
Memory Agent

职责：管理黄金记忆库，检索相关记忆并注入上下文
"""

import logging
from typing import Dict, Any

from src.agents.interfaces import AgentState

logger = logging.getLogger(__name__)


def make_memory_agent(memory_bank, max_memory_chars: int = 500):
    def memory_agent(state: AgentState) -> Dict[str, Any]:
        query = state.get("query", "")

        if memory_bank is None:
            return {"memory_context": ""}

        try:
            memory_context = memory_bank.get_memory_context_prompt(query, top_k=3)
            if not isinstance(memory_context, str):
                memory_context = str(memory_context) if memory_context else ""

            if len(memory_context) > max_memory_chars:
                memory_context = memory_context[:max_memory_chars] + "\n…（更多历史记忆已省略）"

            logger.info("[MemoryAgent] 检索完成: len=%s", len(memory_context))
            return {"memory_context": memory_context}
        except Exception as e:
            logger.error("[MemoryAgent] 检索失败: %s", str(e), exc_info=True)
            return {"memory_context": ""}

    return memory_agent
