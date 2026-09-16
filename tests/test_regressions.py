"""Regression tests use fake models/stores: no network, downloads or user data."""
import copy
from datetime import datetime
import json
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch, Mock
from pathlib import Path
import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from src.documents import chunk_id, read_documents, tokenize
from src.retrievers.hybrid_retriever import HybridRetriever
from src.agents.retrieval_agent import _compute_relevance, make_retrieval_agent
from src.data_ingestion import sync_chunks, split_documents
from src.verifiers.fact_checker import FactChecker, VerificationError
from src.graph.multi_agent_graph import build_multi_agent_rag_graph
from src.runtime import RequestBudget, request_scope, RequestTimeout, bounded_timeout
from src.query import replace_relative_dates, cache_key_for
from src.config import load_config, validate_index


class Store:
    def __init__(self, docs):
        self.docs = docs
    def similarity_search_with_score(self, query, k):
        return [(doc, 0.1) for doc in self.docs[:k]]


def make_retriever(docs, **kwargs):
    return HybridRetriever(Store(docs), BM25Okapi([tokenize(d.page_content) for d in docs]) if docs else None,
                           docs, **kwargs)


class RetrievalTests(unittest.TestCase):
    def test_distinct_chunks_same_source(self):
        docs = [Document(page_content=t, metadata={'source': 'one.txt'}) for t in ['数据库索引', '数据库事务']]
        self.assertNotEqual(chunk_id(docs[0]), chunk_id(docs[1]))
        self.assertEqual(len(make_retriever(docs).retrieve('数据库', 5)), 2)

    def test_nonempty_does_not_mean_relevant(self):
        doc = Document(page_content='香蕉苹果水果')
        self.assertEqual(_compute_relevance('数据库索引优化', [doc]), 0)
        self.assertEqual(make_retriever([doc]).retrieve('数据库索引优化'), [])

    def test_bm25_zero_matches_do_not_add_rrf_votes(self):
        docs = [Document(page_content=t) for t in ['banana', 'apple', 'pear']]
        retriever = make_retriever(docs)
        with patch.object(retriever, 'rank_and_filter', side_effect=lambda q, d, k: d):
            results = retriever.retrieve('database')
        self.assertAlmostEqual(results[0].metadata['rrf_score'], 1 / 61)
        self.assertEqual(len(results), 3)

    def test_duplicate_candidates_vote_once_per_list(self):
        doc = Document(page_content='数据库索引', metadata={'source': 'one.txt'})
        docs = [doc, doc, doc]
        retriever = make_retriever(docs)
        results = retriever.retrieve('数据库')
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0].metadata['rrf_score'], 2 / 61)

    def test_metadata_preserved_when_reading_bm25(self):
        db = Mock()
        db.get.return_value = {'ids': ['a', 'b'], 'documents': ['数据库', '事务'],
                               'metadatas': [{'source': 'a', 'page': 2}, {'source': 'b'}]}
        docs = read_documents(db)
        self.assertEqual(docs[0].metadata['page'], 2)
        self.assertEqual(docs[0].metadata['storage_id'], 'a')

    def test_short_query_does_not_force_hyde(self):
        retriever = make_retriever([Document(page_content='RAG 检索增强生成')])
        hyde = Mock()
        result = make_retrieval_agent(retriever, hyde)({'query': 'RAG'})
        self.assertTrue(result['retrieved_docs'])
        hyde.retrieve.assert_not_called()

    def test_hyde_documents_rescored_against_original_query(self):
        retriever = make_retriever([Document(page_content='香蕉苹果')])
        hyde = Mock()
        hyde.retrieve.return_value = [Document(page_content='无关水果')]
        result = make_retrieval_agent(retriever, hyde)({'query': '数据库'})
        self.assertEqual(result['retrieved_docs'], [])
        hyde.retrieve.assert_called_once()

    def test_reranker_can_reject_every_candidate(self):
        retriever = make_retriever([Document(page_content='数据库')], reranker_model={'fake': True})
        with patch.object(retriever, '_rerank_scores', return_value=[0.01]):
            self.assertEqual(retriever.retrieve('数据库'), [])

    def test_empty_store(self):
        self.assertEqual(make_retriever([]).retrieve('query'), [])


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = 0
    def generate(self, *args, **kwargs):
        self.calls += 1
        return next(self.replies)


def verdict(value):
    return json.dumps([{'claim': '数据库支持事务', 'verdict': value,
                        'evidence': '数据库支持事务'}], ensure_ascii=False)


def graph_for(heavy, light, max_retries=1, docs=True, **kwargs):
    retriever = Mock()
    retriever.retrieve.return_value = [Document(page_content='数据库支持事务', metadata={'source': 'db.txt'})] if docs else []
    checker = FactChecker(light)
    return build_multi_agent_rag_graph(heavy, light, retriever, None, checker, None,
                                     system_identity='Horus', max_retries=max_retries, **kwargs)


class GraphTests(unittest.TestCase):
    def test_cancellation_after_generation_prevents_verification(self):
        budget = RequestBudget(10)
        heavy = Mock()
        def cancel_after_answer(*args, **kwargs):
            budget.cancelled.set()
            return 'answer'
        heavy.generate.side_effect = cancel_after_answer
        light = ScriptedLLM([])
        with self.assertRaises(RequestTimeout):
            with request_scope(budget):
                graph_for(heavy, light).invoke({'query': '数据库'})
        self.assertEqual(light.calls, 0)

    def test_repaired_answer_not_failed_by_history(self):
        heavy = ScriptedLLM(['初始回答'])
        light = ScriptedLLM([verdict('矛盾'), '修正回答', verdict('支持')])
        result = graph_for(heavy, light).invoke({'query': '数据库', 'search_mode': 'local'})
        self.assertEqual(result['answer'], '修正回答')
        self.assertEqual(result['retry_count'], 1)
        self.assertEqual(result['failed_claims'], [])
        self.assertEqual(result['verification_status'], 'passed')
        self.assertEqual(len(result['verification_history']), 2)
        self.assertEqual(light.calls, 3)

    def test_permanent_failure_is_bounded(self):
        heavy = ScriptedLLM(['初始回答'])
        light = ScriptedLLM([verdict('矛盾'), '修正回答', verdict('矛盾')])
        result = graph_for(heavy, light).invoke({'query': '数据库', 'retry_count': 0})
        self.assertEqual(result['retry_count'], 1)
        self.assertEqual(result['verification_status'], 'failed')
        self.assertEqual(light.calls, 3)

    def test_invalid_json_not_passed(self):
        result = graph_for(ScriptedLLM(['回答']), ScriptedLLM(['not-json'])).invoke({'query': '数据库'})
        self.assertEqual(result['verification_status'], 'error')

    def test_no_evidence_no_llm_calls(self):
        heavy, light = ScriptedLLM([]), ScriptedLLM([])
        result = graph_for(heavy, light, docs=False).invoke({'query': '不存在的问题'})
        self.assertIn('无法回答', result['answer'])
        self.assertEqual(heavy.calls + light.calls, 0)

    def test_self_identity_route(self):
        heavy, light = ScriptedLLM(['我是 Horus']), ScriptedLLM([])
        result = graph_for(heavy, light).invoke({'query': '你是谁'})
        self.assertEqual(result['answer'], '我是 Horus')
        self.assertEqual(result['verification_status'], 'skipped')

    def test_no_reference_supports_fabricated_quote(self):
        checker = FactChecker(ScriptedLLM([verdict('支持')]))
        self.assertEqual(checker.check_once('answer', '不同正文')[0]['verdict'], '证据不足')

    def test_context_limit_and_citation_documents_agree(self):
        heavy, light = ScriptedLLM(['answer']), ScriptedLLM([])
        result = graph_for(heavy, light, context_max_chars=80, verification_enabled=False).invoke({'query': '数据库'})
        self.assertLessEqual(len(result['context']), 80)
        for i, doc in enumerate(result['all_docs'], 1):
            self.assertIn(f'[{i}]', result['context'])
            self.assertIn(doc.page_content, result['context'])


class IngestionTests(unittest.TestCase):
    def test_chunks_obey_token_limit_and_have_distinct_ids(self):
        tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text))
        embeddings = SimpleNamespace(client=SimpleNamespace(tokenizer=tokenizer, max_seq_length=20))
        config = {'retrieval': {'chunk_size': 12, 'chunk_overlap': 2}}
        chunks = split_documents([Document(page_content='数据库支持事务。索引可以加速检索。系统支持关键词查询。',
                                           metadata={'source': 'a.txt'})], embeddings, config)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(d.metadata['token_count'] <= 12 for d in chunks))
        self.assertEqual(len({chunk_id(d) for d in chunks}), len(chunks))

    def test_idempotent_sync_and_stale_deletion(self):
        records = {}
        db = Mock()
        db.get.side_effect = lambda: {'ids': list(records)}
        db.add_documents.side_effect = lambda docs, ids: records.update(zip(ids, docs))
        db.delete.side_effect = lambda ids: [records.pop(i) for i in ids]
        docs = [Document(page_content='数据库', metadata={'source': 'a'})]
        self.assertEqual(sync_chunks(db, docs)['added'], 1)
        self.assertEqual(sync_chunks(db, docs)['added'], 0)
        replacement = [Document(page_content='事务', metadata={'source': 'a'})]
        result = sync_chunks(db, replacement)
        self.assertEqual(result, {'added': 1, 'deleted': 1, 'count': 1})

    def test_failed_add_does_not_delete_old(self):
        db = Mock()
        db.get.return_value = {'ids': ['old']}
        db.add_documents.side_effect = RuntimeError('embedding failed')
        with self.assertRaises(RuntimeError):
            sync_chunks(db, [Document(page_content='new')])
        db.delete.assert_not_called()


class RuntimeTests(unittest.TestCase):
    def test_expired_budget_stops_execution(self):
        with self.assertRaises(RequestTimeout):
            with request_scope(RequestBudget(1, started=time.monotonic() - 2)):
                self.fail('must not run')

    def test_timeout_clamped_to_remaining(self):
        with request_scope(RequestBudget(0.5)):
            self.assertLessEqual(bounded_timeout(10), 0.5)

    def test_longest_date_matches_first(self):
        result = replace_relative_dates('大前天、大后天', datetime(2026, 9, 15))
        self.assertEqual(result, '2026年09月12日、2026年09月18日')

    def test_cache_key_accounts_for_history_preferences_index(self):
        original = cache_key_for('q', 'local', [], {}, 'v1', {})
        for args in [('q', 'local', [{'role':'user', 'content':'x'}], {}, 'v1', {}),
                     ('q', 'local', [], {'style':'short'}, 'v1', {}),
                     ('q', 'local', [], {}, 'v2', {})]:
            self.assertNotEqual(original, cache_key_for(*args))


if __name__ == '__main__':
    unittest.main()
