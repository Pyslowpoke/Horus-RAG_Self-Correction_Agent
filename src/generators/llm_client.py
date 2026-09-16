"""One retry layer, per-request budgets, bounded output and provider fallback."""
import logging
import time
import openai
from src.runtime import bounded_timeout, check_budget, record_usage, RequestTimeout, emit_event

logger = logging.getLogger(__name__)


class FaultTolerantLLM:
    def __init__(self, primary_config, fallback_config, max_retries=0, max_tokens=512, temperature=0.1):
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.providers = []
        seen = set()
        for config in (primary_config, fallback_config):
            identity = (config['api_base'], config['model'])
            if not config.get('api_key') or identity in seen:
                continue
            seen.add(identity)
            client = openai.OpenAI(api_key=config['api_key'], base_url=config['api_base'], max_retries=0)
            self.providers.append((client, config))
        if not self.providers:
            raise ValueError('未配置可用的 LLM API Key')

    def generate(self, messages, temperature=None, timeout=None, max_tokens=None, stream=False):
        last_error = None
        for client, config in self.providers:
            for attempt in range(self.max_retries + 1):
                check_budget()
                try:
                    record_usage()
                    if stream:
                        emit_event('draft_reset')
                    stream_options = {'stream': True, 'stream_options': {'include_usage': True}} if stream else {}
                    response = client.chat.completions.create(model=config['model'], messages=messages,
                        temperature=self.temperature if temperature is None else temperature,
                        max_tokens=max_tokens or self.max_tokens,
                        timeout=bounded_timeout(timeout or config.get('timeout', 15)), **stream_options)
                    if stream:
                        parts = []
                        try:
                            for chunk in response:
                                check_budget()
                                record_usage(chunk)
                                text = chunk.choices[0].delta.content if chunk.choices else None
                                if text:
                                    parts.append(text)
                                    emit_event('token', text)
                        finally:
                            response.close()
                        check_budget()
                        return ''.join(parts)
                    record_usage(response)
                    check_budget()
                    return response.choices[0].message.content or ''
                except RequestTimeout:
                    raise
                except (openai.APIConnectionError, openai.APIStatusError) as error:
                    last_error = error
                    status = getattr(error, 'status_code', None)
                    retryable = status is None or status in (408, 409, 429) or status >= 500
                    logger.warning('LLM 调用失败: model=%s status=%s attempt=%s', config['model'], status, attempt + 1)
                    if not retryable:
                        break
                    if attempt < self.max_retries:
                        time.sleep(bounded_timeout(min(2 ** attempt, 2)))
        check_budget()
        raise RuntimeError('所有 LLM 链路均不可用，请检查配置及网络') from last_error
