import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
import openai
import httpx
from src.generators.llm_client import FaultTolerantLLM
from src.runtime import RequestBudget, request_scope


class LLMTests(unittest.TestCase):
    def configs(self):
        return ({'api_key': 'test', 'api_base': 'https://primary.invalid', 'model': 'primary', 'timeout': 10},
                {'api_key': 'test', 'api_base': 'https://fallback.invalid', 'model': 'fallback', 'timeout': 10})

    def test_sdk_retry_disabled_and_output_bounded(self):
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='ok'))],
                                   usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2))
        client = Mock()
        client.chat.completions.create.return_value = response
        with patch('openai.OpenAI', return_value=client) as constructor:
            llm = FaultTolerantLLM(*self.configs(), max_tokens=123)
            with request_scope(RequestBudget(1)) as budget:
                self.assertEqual(llm.generate([]), 'ok')
            self.assertTrue(all(call.kwargs['max_retries'] == 0 for call in constructor.call_args_list))
            args = client.chat.completions.create.call_args.kwargs
            self.assertEqual(args['max_tokens'], 123)
            self.assertLessEqual(args['timeout'], 1)
            self.assertEqual(budget.metrics['llm_calls'], 1)
            self.assertEqual(budget.metrics['input_tokens'], 5)

    def test_failure_switches_provider_once(self):
        primary, fallback = Mock(), Mock()
        primary.chat.completions.create.side_effect = openai.APIConnectionError(request=httpx.Request('POST', 'https://primary.invalid'))
        fallback.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='fallback'))], usage=None)
        with patch('openai.OpenAI', side_effect=[primary, fallback]):
            llm = FaultTolerantLLM(*self.configs())
            self.assertEqual(llm.generate([]), 'fallback')
        self.assertEqual(primary.chat.completions.create.call_count, 1)
        self.assertEqual(fallback.chat.completions.create.call_count, 1)

    def test_streaming_closes_stream_and_emits_draft(self):
        class Stream:
            closed = False
            def __iter__(self):
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content='hello'))], usage=None)
                yield SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=4, completion_tokens=1))
            def close(self):
                self.closed = True
        stream = Stream()
        client = Mock()
        client.chat.completions.create.return_value = stream
        with patch('openai.OpenAI', return_value=client):
            llm = FaultTolerantLLM(*self.configs())
            with request_scope(RequestBudget(1)) as budget:
                self.assertEqual(llm.generate([], stream=True), 'hello')
            self.assertTrue(stream.closed)
            self.assertEqual(budget.metrics['output_tokens'], 1)
            self.assertIn('first_token_seconds', budget.metrics)
            self.assertEqual(budget.events.get_nowait(), ('draft_reset', ''))
            self.assertEqual(budget.events.get_nowait(), ('token', 'hello'))

    def test_same_provider_not_called_twice_as_fallback(self):
        primary, _ = self.configs()
        with patch('openai.OpenAI') as constructor:
            llm = FaultTolerantLLM(primary, primary)
            self.assertEqual(len(llm.providers), 1)
            self.assertEqual(constructor.call_count, 1)
