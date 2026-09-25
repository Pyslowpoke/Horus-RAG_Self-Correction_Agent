"""Evaluation-metric contract tests, separate from product's 36 regressions."""
import unittest
from langchain_core.documents import Document
from run_local import score

class MetricsTests(unittest.TestCase):
    def test_multiple_evidence_requires_both(self):
        c={'evidence':[{'source':'a','quote':'600元'},{'source':'b','quote':'800元'}]}
        s=score(c,[Document(page_content='上限600元',metadata={'source':'a'})],None)
        self.assertEqual((s['any_hit'],s['all_hit'],s['evidence_recall']),(1,0,0.5))
    def test_wrong_source_does_not_count(self):
        c={'evidence':[{'source':'a','quote':'600元'}]}
        self.assertEqual(score(c,[Document(page_content='600元',metadata={'source':'b'})],None)['any_hit'],0)
    def test_failed_query_is_not_correct_abstention(self):
        self.assertEqual(score({'evidence':[]},[],'TimeoutError')['empty_success'],0)
    def test_empty_success_is_not_failure(self):
        self.assertEqual(score({'evidence':[]},[],None)['empty_success'],1)
    def test_precision_denominators_differ(self):
        c={'evidence':[{'source':'a','quote':'600元'}]}
        s=score(c,[Document(page_content='600元',metadata={'source':'a'})],None)
        self.assertEqual((s['precision_at_5'],s['returned_precision']),(0.2,1))
    def test_whitespace_normalization(self):
        c={'evidence':[{'source':'a','quote':'600 元'}]}
        self.assertEqual(score(c,[Document(page_content='600\n元',metadata={'source':'a'})],None)['any_hit'],1)

if __name__=='__main__':unittest.main()
