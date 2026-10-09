import unittest
from unittest.mock import Mock
from langchain_core.documents import Document
from src.graph.multi_agent_graph import build_multi_agent_rag_graph
from src.agents.fact_check_agent import make_fact_check_agent
from src.retrievers.hybrid_retriever import HybridRetriever

class EvidenceTests(unittest.TestCase):
 def test_generation_failure_preserves_evidence_without_claiming_success(self):
  from src.agents.generation_agent import make_generation_agent
  llm=Mock();llm.generate.side_effect=RuntimeError('offline')
  doc=Document(page_content='支持 TXT 和 PDF',metadata={'source':'guide'})
  result=make_generation_agent(llm)({'query':'文件格式','search_mode':'local','context':doc.page_content,'all_docs':[doc]})
  self.assertTrue(result['generation_error']);self.assertIn('TXT 和 PDF',result['answer']);self.assertIn('未生成结论',result['answer'])
 def test_refusal_is_not_verification_success(self):
  checker=Mock()
  result=make_fact_check_agent(checker)({'answer':'缺少依据，无法回答。','all_docs':[Document(page_content='介绍')],'context':'介绍'})
  self.assertEqual(result['verification_status'],'unanswered');checker.check_once.assert_not_called()
 def test_hybrid_retains_local_and_web_evidence(self):
  llm=Mock();llm.generate.return_value='文件格式包括TXT和PDF [1]；新公告 [2]'
  local=Mock();local.retrieve.return_value=[Document(page_content='文件格式 TXT PDF',metadata={'source':'local.txt'})]
  local.rank_and_filter.side_effect=lambda q,d,k:d
  web=Mock();web.retrieve.return_value=[Document(page_content='文件格式新公告',metadata={'source':'web','url':'https://example.org/new'})]
  graph=build_multi_agent_rag_graph(llm,llm,local,web,Mock(),None,verification_enabled=False,system_identity='Horus')
  result=graph.invoke({'query':'文件格式','search_mode':'hybrid'})
  web.retrieve.assert_called_once();self.assertEqual(len(result['all_docs']),2)
 def test_retrieval_adds_adjacent_section(self):
  a=Document(page_content='产品概述',metadata={'source':'guide','chunk_index':0,'chunk_id':'a'})
  b=Document(page_content='文件格式 TXT PDF',metadata={'source':'guide','chunk_index':1,'chunk_id':'b'})
  db=Mock();db.similarity_search_with_score.return_value=[(a,.1)]
  r=HybridRetriever(db,None,[a,b]);r.rank_and_filter=Mock(return_value=[a])
  self.assertEqual([d.page_content for d in r.retrieve('产品文件格式')],['产品概述','文件格式 TXT PDF'])
