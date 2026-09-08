"""
事实核查器

职责：核查 + 重写循环（不重新生成）
"""

from src.generators.prompt_templates import FACT_CHECK_PROMPT, REWRITE_RESPONSE_PROMPT
import json
import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class FactChecker:
    def __init__(self, llm_client, retriever=None, max_retries: int = 2, system_identity: str = ""):
        self.llm_client = llm_client
        self.retriever = retriever
        self.max_retries = max_retries
        self.system_identity = system_identity

    def _parse_json_response(self, raw: str) -> List[Dict[str, str]]:
        """安全解析 LLM 返回的 JSON，支持 markdown 代码块"""
        if not raw:
            return []

        # 直接解析
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict) and "claims" in parsed:
                return parsed["claims"]
        except json.JSONDecodeError:
            pass

        # 从 markdown 代码块提取
        m = re.search(r'```json\s*(\[.*?\])\s*```', raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        # 搜索数组
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass

        logger.error("[FactChecker] JSON 解析失败: %s", raw[:200])
        return []

    def _validate_log_entry(self, entry: Dict) -> Dict[str, str]:
        """规范化单条核查日志"""
        if not isinstance(entry, dict):
            return {"claim": "未提取到陈述", "verdict": "证据不足", "evidence": ""}
        
        claim = entry.get("claim") or entry.get("statement") or "未提取到陈述"
        verdict = entry.get("verdict", "")
        if verdict not in ("支持", "矛盾", "证据不足"):
            if "支持" in verdict or "正确" in verdict:
                verdict = "支持"
            elif "矛盾" in verdict or "错误" in verdict:
                verdict = "矛盾"
            else:
                verdict = "证据不足"
        evidence = entry.get("evidence", "") or ""
        return {"claim": claim, "verdict": verdict, "evidence": evidence}

    def _fact_check(self, answer: str, context: str) -> List[Dict[str, str]]:
        """核查回答中的陈述"""
        check_result = self.llm_client.generate([
            {"role": "user", "content": FACT_CHECK_PROMPT.format(response=answer, context=context)}
        ])
        raw_claims = self._parse_json_response(check_result)
        return [self._validate_log_entry(entry) for entry in raw_claims]

    def _rewrite(self, answer: str, failed_claims: List[Dict], context: str) -> str:
        """修正有问题的陈述"""
        failed_json = json.dumps(failed_claims, ensure_ascii=False, indent=2)
        return self.llm_client.generate([
            {"role": "user", "content": REWRITE_RESPONSE_PROMPT.format(
                original_response=answer, failed_claims=failed_json, context=context
            )}
        ])

    def execute_pipeline(self, query: str, initial_answer: str, context: str) -> Dict[str, Any]:
        """
        执行核查 + 重写流水线（不重新生成）

        参数:
            query: 用户原始问题
            initial_answer: generation_agent 生成的初始回答
            context: 检索到的参考资料

        返回:
            dict: {
                "final_answer": str,
                "verification_log": list,
                "retry_count": int
            }
        """
        current_answer = initial_answer
        retry_count = 0
        verification_log = []

        for attempt in range(self.max_retries + 1):
            # 1. 核查
            checks = self._fact_check(current_answer, context)
            verification_log.extend(checks)
            failed = [c for c in checks if c.get("verdict") in ("矛盾", "证据不足")]

            # 2. 如果没有失败，直接返回
            if not failed:
                logger.info("[FactChecker] 核查通过，attempt=%s", attempt)
                return {
                    "final_answer": current_answer,
                    "verification_log": verification_log,
                    "retry_count": retry_count,
                }

            # 3. 如果有失败且还有重试机会，重写
            if attempt < self.max_retries:
                logger.info("[FactChecker] 重写 %s/%s，失败数: %s", attempt + 1, self.max_retries, len(failed))
                current_answer = self._rewrite(current_answer, failed, context)
                retry_count += 1
            else:
                # 达到最大重试次数，返回最后一次的结果
                logger.info("[FactChecker] 达到最大重试次数，返回当前结果")
                return {
                    "final_answer": current_answer,
                    "verification_log": verification_log,
                    "retry_count": retry_count,
                }

        return {
            "final_answer": current_answer,
            "verification_log": verification_log,
            "retry_count": retry_count,
        }
