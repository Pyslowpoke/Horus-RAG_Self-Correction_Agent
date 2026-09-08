"""
双路容灾 LLM 客户端

功能：
1. 主链路使用 DeepSeek API，备用链路使用 SiliconFlow API
2. 主链路失败后自动重试，重试耗尽后切换到备用链路
3. 记录累计 Token 消耗，供后续成本评估
"""

import openai
import logging
import time
from typing import List, Dict, Any, Optional

# 配置日志：输出到控制台，级别为 INFO
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaultTolerantLLM:
    """
    容灾大模型客户端

    用法：
        config = {
            "api_key": "sk-xxx",
            "api_base": "https://api.deepseek.com/v1",
            "model": "deepseek-chat"
        }
        llm = FaultTolerantLLM(primary_config=config, fallback_config=config2)
        answer = llm.generate([{"role": "user", "content": "你好"}])
        print(llm.get_token_usage())  # 查看 token 消耗
    """

    def __init__(
            self,
            primary_config: Dict[str, Any],
            fallback_config: Dict[str, Any],
            max_retries: int = 2,
    ):
        """
        初始化两个 OpenAI 兼容客户端

        参数:
            primary_config: 主链路配置，必须包含 api_key, api_base, model
            fallback_config: 备用链路配置，结构同上
            max_retries: 单条链路内部重试次数（不含首次调用）
        """
        # ----- 主链路客户端 -----
        self.primary_client = openai.OpenAI(
            api_key=primary_config["api_key"],
            base_url=primary_config["api_base"],
        )
        self.primary_model = primary_config["model"]

        # ----- 备用链路客户端 -----
        self.fallback_client = openai.OpenAI(
            api_key=fallback_config["api_key"],
            base_url=fallback_config["api_base"],
        )
        self.fallback_model = fallback_config["model"]

        # 重试次数 & Token 累计器
        self.max_retries = max_retries
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _call_with_retry(
            self,
            client: openai.OpenAI,
            model: str,
            messages: List[Dict[str, str]],
            temperature: float,
            timeout: int,
    ) -> str:
        """
        单条链路调用：带重试、Token 统计、异常分类

        参数:
            client: OpenAI 客户端实例
            model: 模型名称
            messages: 对话消息列表 [{"role": "user", "content": "..."}]
            temperature: 生成温度
            timeout: 超时秒数

        返回:
            str: 模型生成的文本

        异常:
            仅当所有重试都失败时，抛出最后一次捕获的异常
        """
        last_exception = None

        # 总共尝试 max_retries + 1 次（首次 + 重试）
        for attempt in range(self.max_retries + 1):
            try:
                # 发起 API 调用
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    timeout=timeout,
                )

                # ----- 累计 Token 消耗 -----
                if hasattr(response, "usage") and response.usage:
                    self.total_input_tokens += response.usage.prompt_tokens
                    self.total_output_tokens += response.usage.completion_tokens

                # 返回生成的文本
                return response.choices[0].message.content

            except (
                    openai.APIError,
                    openai.APIConnectionError,
                    openai.RateLimitError,
                    openai.APITimeoutError,
            ) as e:
                # 仅捕获网络/服务端异常，不捕获代码错误（如 KeyError）
                last_exception = e
                logger.warning(
                    f"API 调用失败 (尝试 {attempt + 1}/{self.max_retries + 1}): {e}"
                )

                # 如果还有重试机会，退避等待后继续
                if attempt < self.max_retries:
                    sleep_sec = 1 * (attempt + 1)  # 退避策略：1秒、2秒、3秒
                    logger.info(f"等待 {sleep_sec} 秒后重试...")
                    time.sleep(sleep_sec)
                continue

        # 所有重试均失败，抛出最后一次捕获的异常
        raise last_exception

    def generate(
            self,
            messages: List[Dict[str, str]],
            temperature: float = 0.1,
            timeout: int = 10,
    ) -> str:
        """
        生成回答：优先主链路，彻底失败后切备用链路

        参数:
            messages: 对话消息列表
            temperature: 生成温度（默认 0.1，保守）
            timeout: 超时秒数（默认 10）

        返回:
            str: 最终生成的文本

        异常:
            如果主备两条链路均不可用，抛出 RuntimeError
        """
        # ----- 第一步：主链路 -----
        try:
            logger.info("正在调用主链路 (DeepSeek)...")
            return self._call_with_retry(
                client=self.primary_client,
                model=self.primary_model,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
            )
        except Exception as e:
            # _call_with_retry 内部已重试，到这里说明主链路彻底不可用
            logger.warning(f"主链路彻底失败: {e}，切换到备用链路")

        # ----- 第二步：备用链路 -----
        try:
            logger.info("正在调用备用链路 (SiliconFlow)...")
            return self._call_with_retry(
                client=self.fallback_client,
                model=self.fallback_model,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
            )
        except Exception as e:
            # 备用链路也失败，程序无法继续
            logger.error(f"备用链路也失败了: {e}")
            raise RuntimeError("所有 LLM 链路均不可用，请检查 API 配置和网络") from e

    def get_token_usage(self) -> dict:
        """
        获取累计 Token 消耗

        返回:
            dict: {"input_tokens": int, "output_tokens": int}
        """
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
        }



