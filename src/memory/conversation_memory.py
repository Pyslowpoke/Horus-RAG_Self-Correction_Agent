"""
会话记忆模块

存储对话历史，支持多轮对话上下文。
"""

from typing import List, Dict
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Message:
    role: str           # "user" 或 "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    metadata: dict = field(default_factory=dict)


class ConversationMemory:
    """
    用法：
        memory = ConversationMemory(max_history=20)
        memory.add_user_message("什么是RAG？")
        memory.add_assistant_message("RAG是检索增强生成技术...")
        history = memory.get_history()
    """

    def __init__(self, max_history: int = 20):
        self.history: List[Message] = []
        self.max_history = max_history

    def add_user_message(self, content: str, metadata: dict = None):
        self.history.append(Message(role="user", content=content, metadata=metadata or {}))
        self._trim()

    def add_assistant_message(self, content: str, metadata: dict = None):
        self.history.append(Message(role="assistant", content=content, metadata=metadata or {}))
        self._trim()

    def get_history(self) -> List[Dict[str, str]]:
        """返回 LLM 兼容的消息列表"""
        return [{"role": m.role, "content": m.content} for m in self.history]

    def get_recent_context(self, n: int = 5) -> str:
        """获取最近 n 轮对话的格式化字符串"""
        recent = self.history[-(n * 2):]
        return "\n".join(
            f"{'用户' if m.role == 'user' else '助手'}: {m.content}" for m in recent
        )

    def clear(self):
        self.history.clear()

    def _trim(self):
        while len(self.history) > self.max_history:
            self.history.pop(0)

    def __len__(self):
        return len(self.history)
