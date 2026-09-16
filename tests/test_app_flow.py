"""Front-end flow smoke test, with all model/network work replaced by fakes."""
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest
from src.runtime import emit_event


class AppFlowTests(unittest.TestCase):
    def test_identity_question_skips_local_model_loading(self):
        import streamlit as st
        st.cache_resource.clear()
        llm = Mock()
        llm.generate.return_value = '我是 Horus'
        with patch('src.components.make_llm', return_value=llm), \
             patch('src.components.load_embeddings', side_effect=AssertionError('must not load')):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=30)
            app.session_state['query'] = '你是谁'
            app.session_state['page'] = 'processing'
            app.run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['result']['answer'], '我是 Horus')
            self.assertEqual(llm.generate.call_count, 1)
        st.cache_resource.clear()

    def test_local_request_runs_in_worker_and_renders_final_answer(self):
        import streamlit as st
        st.cache_resource.clear()
        heavy = Mock()
        def answer(*args, **kwargs):
            emit_event('token', '测试回答 [1]')
            return '测试回答 [1]'
        heavy.generate.side_effect = answer
        light_client = Mock()
        light_client.generate.return_value = '[{"claim":"测试事实","verdict":"支持","evidence":"测试事实"}]'
        retriever = Mock()
        retriever.reranker_model = None
        retriever.retrieve.return_value = [Document(page_content='测试事实', metadata={'source':'test.txt'})]
        bank = Mock()
        bank.get_memory_context_prompt.return_value = ''
        with patch('src.components.make_llm', side_effect=lambda c, light=False: light_client if light else heavy), \
             patch('src.components.load_embeddings', return_value=Mock()), \
             patch('src.components.load_db', return_value=Mock()), \
             patch('src.components.build_bm25', return_value=(None, [])), \
             patch('src.components.load_reranker', return_value=None), \
             patch('src.components.make_retriever', return_value=retriever), \
             patch('src.memory.gold_memory_bank.GoldMemoryBank', return_value=bank), \
             patch('src.memory.gold_memory_bank.import_all_corrections'), \
             patch('src.memory.gold_memory_bank.start_watcher'):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=30)
            self.assertFalse(app.exception)
            app.session_state['query'] = '测试问题'
            app.session_state['page'] = 'processing'
            app.session_state['use_hybrid'] = False
            app.session_state['use_web_search'] = False
            app.run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['result']['answer'], '测试回答 [1]')
            self.assertEqual(app.session_state['result']['verification_status'], 'passed')
            self.assertIn('metrics', app.session_state['result'])
        st.cache_resource.clear()
