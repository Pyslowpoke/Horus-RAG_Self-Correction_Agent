"""
互联网搜索检索器（百度千帆 API）

环境变量：BAIDU_API_KEY
"""

import os
import time
import logging
import requests
from typing import List, Optional, Dict
from langchain_core.documents import Document
from src.runtime import bounded_timeout, check_budget

logger = logging.getLogger(__name__)


class WebSearchRetriever:
    """互联网搜索检索器（百度千帆）"""

    def __init__(self, max_results: int = 5, timeout: int = 8, max_content_chars: int = 600):
        self.max_results = max_results
        self.timeout = timeout
        self.max_content_chars = max_content_chars

        self.baidu_api_key = os.getenv("BAIDU_API_KEY")
        if not self.baidu_api_key:
            raise ValueError(
                "未找到 BAIDU_API_KEY，请在 .env 文件中配置。\n"
                "获取方式：百度千帆控制台 → 应用接入 → 创建应用 → 获取 API Key。"
            )

        logger.info("[WebSearch] 百度千帆搜索已初始化 (超时=%ss, 结果数=%s)", timeout, max_results)

    def _search_baidu(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """调用百度千帆智能搜索 API"""
        url = "https://qianfan.baidubce.com/v2/ai_search/chat/completions"
        headers = {
            "X-Appbuilder-Authorization": f"Bearer {self.baidu_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "messages": [{"role": "user", "content": query}],
            "resource_type_filter": [{"type": "web", "top_k": max_results}],
            "search_source": "baidu_search_v2",
        }

        last_exception = None
        for attempt in range(2):
            check_budget()
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=bounded_timeout(self.timeout))
                check_budget()

                resp.raise_for_status()
                data = resp.json()

                if data.get("code") not in (None, 0):
                    logger.warning("[百度搜索] 业务错误: %s", data.get("message"))
                    raise ValueError('搜索服务返回业务错误')

                references = data.get("references", [])
                if not references:
                    return []

                parsed = []
                for item in references:
                    content = (item.get("content", "") or item.get("summary", "")).strip()
                    if len(content) > self.max_content_chars:
                        content = content[:self.max_content_chars] + "..."
                    parsed.append({
                        "title": item.get("title", "无标题"),
                        "href": item.get("url", ""),
                        "body": content,
                    })
                return parsed

            except requests.exceptions.RequestException as e:
                last_exception = e
                logger.warning("[百度搜索] 尝试 %s/2 失败: %s", attempt + 1, e)
                if isinstance(e, requests.HTTPError) and e.response is not None and 400 <= e.response.status_code < 500 and e.response.status_code not in (408, 429):
                    raise
                if attempt == 0:
                    time.sleep(bounded_timeout(1))

        logger.error("[百度搜索] 所有重试失败: %s", last_exception)
        raise RuntimeError('联网搜索请求失败') from last_exception

    def retrieve(self, query: str, max_results: Optional[int] = None) -> List[Document]:
        """执行互联网搜索，返回 Document 列表"""
        n = max_results or self.max_results
        raw_results = self._search_baidu(query, n)

        if not raw_results:
            logger.warning("[WebSearch] 搜索无结果")
            return []

        docs = []
        for i, item in enumerate(raw_results):
            title = item.get("title", "无标题")
            body = item.get("body", "")
            url = item.get("href", "")
            content = f"标题：{title}\n摘要：{body}" if body else title
            metadata = {"source": "web_search (Baidu)", "url": url, "title": title, "rank": i + 1}
            docs.append(Document(page_content=content, metadata=metadata))

        return docs

    def retrieve_as_context(self, query: str, max_results: Optional[int] = None) -> str:
        """搜索并返回格式化上下文字符串"""
        docs = self.retrieve(query, max_results)
        if not docs:
            return "（互联网搜索未返回相关结果）"

        parts = []
        for i, doc in enumerate(docs):
            url = doc.metadata.get("url", "无链接")
            parts.append(f"[{i+1}] {doc.page_content[:300]}\n来源: {url}")
        return "\n\n".join(parts)
