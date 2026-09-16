"""Chinese embedding contracts: query-only instruction, normalization and isolation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from src.config import load_config, embedding_signature, validate_index
from src.components import load_embeddings
from src.memory.gold_memory_bank import GoldMemoryBank
from langchain_core.documents import Document


class ChineseEmbeddingTests(unittest.TestCase):
    def test_instruction_only_applies_to_queries(self):
        config=load_config()
        config['embedding'].update(model_name='BAAI/bge-small-zh-v1.5', normalize_embeddings=True,
            query_prefix='为这个句子生成表示以用于检索相关文章：')
        fake=Mock()
        fake.embed_documents.return_value=[[1.,0.]]
        fake.embed_query.return_value=[1.,0.]
        with patch('langchain_community.embeddings.HuggingFaceEmbeddings', return_value=fake) as constructor:
            embedding=load_embeddings(config)
            embedding.embed_documents(['文档正文'])
            embedding.embed_query('用户问题')
        fake.embed_documents.assert_called_once_with(['文档正文'])
        fake.embed_query.assert_called_once_with('为这个句子生成表示以用于检索相关文章：用户问题')
        self.assertTrue(constructor.call_args.kwargs['encode_kwargs']['normalize_embeddings'])
        self.assertIs(embedding.client, fake.client)

    def test_old_index_rejected_by_new_embedding(self):
        config=load_config()
        old=copy.deepcopy(config)
        old['embedding'].update(model_name='all-MiniLM-L6-v2',normalize_embeddings=False,query_prefix='')
        config['embedding'].update(model_name='BAAI/bge-small-zh-v1.5',normalize_embeddings=True)
        with tempfile.TemporaryDirectory() as directory:
            config['db']['chroma_persist_dir']=directory
            Path(directory,'index_manifest.json').write_text(json.dumps({'embedding':embedding_signature(old)}),encoding='utf-8')
            with self.assertRaises(ValueError):
                validate_index(config)

    def test_memory_threshold_uses_model_specific_setting(self):
        with tempfile.TemporaryDirectory() as directory, patch('src.memory.gold_memory_bank.Chroma') as factory:
            bank=GoldMemoryBank(directory,embeddings=Mock(),distance_threshold=0.35)
            factory.return_value.similarity_search_with_score.return_value=[
                (Document(page_content='相关',metadata={'mem_id':'a'}),0.2),
                (Document(page_content='弱相关',metadata={'mem_id':'b'}),0.5)]
            self.assertEqual([r['mem_id'] for r in bank.retrieve_preferences('问题')],['a'])
            self.assertEqual(len(bank.retrieve_preferences('问题',distance_threshold=0.6)),2)
