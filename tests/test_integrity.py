import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from langchain_core.documents import Document
from src.citations import audit_citations
from src.graph.multi_agent_graph import build_multi_agent_rag_graph
from src.memory.gold_memory_bank import GoldMemoryBank, CorrectionFileHandler
from src.verifiers.fact_checker import FactChecker, VerificationError


class IntegrityTests(unittest.TestCase):
    def test_reference_groups_and_non_references(self):
        result = audit_citations('结论 [1, 2]；其他 [0] [9]。链接 [8](https://example.org) ```[7]```', 2)
        self.assertEqual(result['used_ids'], [0, 1, 2, 9])
        self.assertEqual(result['invalid_ids'], [0, 9])
        self.assertEqual(audit_citations('无引用', 2)['status'], 'missing')

    def test_graph_checks_references_when_model_verification_disabled(self):
        llm = Mock(); llm.generate.return_value = '支持 TXT [7]'
        retriever = Mock(); retriever.retrieve.return_value = [Document(page_content='支持 TXT', metadata={'source': 'guide'})]
        graph = build_multi_agent_rag_graph(llm, llm, retriever, None, Mock(), None, verification_enabled=False)
        result = graph.invoke({'query': '支持什么格式', 'search_mode': 'local'})
        self.assertEqual(result['citation_audit']['invalid_ids'], [7])
        self.assertEqual(result['verification_status'], 'disabled')

    def test_invalid_reference_cannot_be_labelled_verified(self):
        llm = Mock(); llm.generate.return_value = '支持 TXT [7]'
        retriever = Mock(); retriever.retrieve.return_value = [Document(page_content='支持 TXT', metadata={'source': 'guide'})]
        checker = Mock(); checker.check_once.return_value = [{'claim': '支持 TXT', 'verdict': '支持', 'evidence': '支持 TXT'}]
        graph = build_multi_agent_rag_graph(llm, llm, retriever, None, checker, None)
        result = graph.invoke({'query': '支持什么格式', 'search_mode': 'local'})
        self.assertEqual(result['verification_status'], 'failed')

    def test_ui_displays_reference_warning(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'))
        app.session_state['page'] = 'result'
        app.session_state['result'] = {'answer': '答案 [9]', 'citation_audit': {'status': 'invalid', 'invalid_ids': [9]},
                                       'verification_status': 'disabled'}
        app.run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(any('不存在的证据编号' in w.value for w in app.warning))

    def test_memory_csv_preserves_text_and_does_not_dedupe_prefixes(self):
        with tempfile.TemporaryDirectory() as directory, patch('src.memory.gold_memory_bank.Chroma'):
            bank = GoldMemoryBank(directory, embeddings=Mock())
            query = '同一个很长的提问开头需要超过二十个字符然后还有细节,A'
            answer = '答案含逗号,换行\n以及"引号"'
            bank.add_memory(query, '旧答案', answer, source_file='a.txt')
            with bank.records_file.open(encoding='utf-8', newline='') as f:
                row = next(csv.DictReader(f))
            self.assertEqual(row['correct_answer'], answer)
            self.assertEqual(bank.get_memory_count(), 1)
            self.assertTrue(bank._is_duplicate('a.txt', query, answer))
            self.assertFalse(bank._is_duplicate('a.txt', query + '不同', answer))
            self.assertFalse(bank._is_duplicate('b.txt', query, answer))
            self.assertFalse(bank._is_duplicate('a.txt', query, '更新的修正'))

    def test_incomplete_file_can_be_completed_after_creation(self):
        from watchdog.events import FileCreatedEvent, FileModifiedEvent
        with tempfile.TemporaryDirectory() as directory, patch('src.memory.gold_memory_bank.Chroma'):
            bank = GoldMemoryBank(directory, embeddings=Mock())
            handler = CorrectionFileHandler(bank)
            file = Path(directory, 'input.txt')
            file.write_text('Q: 问题\n', encoding='utf-8')
            handler.on_created(FileCreatedEvent(str(file)))
            file.write_text('Q: 问题\nCorrect: 正确答案\n', encoding='utf-8')
            handler.on_modified(FileModifiedEvent(str(file)))
            handler.on_modified(FileModifiedEvent(str(file)))
            self.assertEqual(bank.get_memory_count(), 1)

    def test_empty_checker_compatibility_api_is_not_success(self):
        llm = Mock(); llm.generate.return_value = '{"claims": []}'
        result = FactChecker(llm).execute_pipeline('问题', '回答', '资料')
        self.assertEqual(result['verification_status'], 'no_claims')

    def test_duplicate_checker_claims_fail_closed(self):
        llm = Mock(); llm.generate.return_value = '{"claims": [{"claim":"事实","verdict":"支持","evidence":"资料"},{"claim":"事实","verdict":"支持","evidence":"资料"}]}'
        with self.assertRaises(VerificationError):
            FactChecker(llm).check_once('回答', '资料')

    def test_compatibility_api_rejects_unverified_rewrite(self):
        llm = Mock()
        failed = '{"claims":[{"claim":"事实","verdict":"证据不足","evidence":""}]}'
        llm.generate.side_effect = [failed, '新答案', failed]
        result = FactChecker(llm).execute_pipeline('问题', '原答案', '资料')
        self.assertEqual(result['final_answer'], '原答案')
        self.assertTrue(result['rewrite_rejected'])
