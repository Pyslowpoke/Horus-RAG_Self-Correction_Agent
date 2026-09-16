"""Offline checks for web evidence filtering and failure handling."""
import unittest
from unittest.mock import Mock
from langchain_core.documents import Document
from src.agents.web_search_agent import make_web_search_agent
from src.runtime import RequestTimeout


class WebTests(unittest.TestCase):
    def test_filters_unrelated_web_result(self):
        web = Mock()
        web.retrieve.return_value = [Document(page_content='banana apple'), Document(page_content='database indexing')]
        result = make_web_search_agent(web)({'query':'database', 'top_k':5})
        self.assertEqual(len(result['web_docs']), 1)
        self.assertEqual(result['web_docs'][0].page_content, 'database indexing')

    def test_web_failure_is_not_empty_success(self):
        web = Mock()
        web.retrieve.side_effect = RuntimeError('offline')
        with self.assertLogs('src.agents.web_search_agent', level='ERROR'):
            result = make_web_search_agent(web)({'query':'database'})
        self.assertIn('web_error', result)

    def test_budget_timeout_is_not_swallowed(self):
        web = Mock()
        web.retrieve.side_effect = RequestTimeout()
        with self.assertRaises(RequestTimeout):
            make_web_search_agent(web)({'query':'database'})
