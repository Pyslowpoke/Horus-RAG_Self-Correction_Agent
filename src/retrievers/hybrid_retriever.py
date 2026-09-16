"""Hybrid retrieval with stable identity, observable scores and evidence filtering."""
import logging
import numpy as np
from langchain_core.documents import Document
from src.documents import chunk_id, unique_docs, tokenize, lexical_relevance
from src.runtime import timed, check_budget, RequestTimeout

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self, chroma_db, bm25_index, all_docs, reranker_model=None,
                 candidate_k=20, rerank_k=10, rrf_k=60, min_lexical_score=0.3,
                 max_vector_distance=None, min_rerank_score=0.3, **kwargs):
        self.chroma_db = chroma_db
        self.bm25_index = bm25_index
        self.all_docs = all_docs
        self.reranker_model = reranker_model
        self.candidate_k = candidate_k
        self.rerank_k = rerank_k
        self.rrf_k = rrf_k
        self.min_lexical_score = min_lexical_score
        self.max_vector_distance = max_vector_distance
        self.min_rerank_score = min_rerank_score
        if bm25_index is not None and len(all_docs) != bm25_index.corpus_size:
            raise ValueError('BM25 文档数与索引不一致')
        self.doc_terms = [set(tokenize(doc.page_content)) for doc in all_docs]

    def _get_doc_id(self, doc):
        return chunk_id(doc)

    def retrieve(self, query, top_k=5, extra_docs=None):
        if not query.strip() or top_k < 1 or not self.all_docs:
            return []
        candidate_k = min(self.candidate_k, len(self.all_docs))
        with timed('vector_search'):
            vector_results = []
            for doc, distance in self.chroma_db.similarity_search_with_score(query, k=candidate_k):
                if self.max_vector_distance is None or distance <= self.max_vector_distance:
                    vector_results.append(Document(page_content=doc.page_content,
                        metadata={**doc.metadata, 'vector_distance': float(distance)}))
        with timed('bm25_search'):
            tokens = tokenize(query)
            terms = set(tokens)
            scores = self.bm25_index.get_scores(tokens) if self.bm25_index is not None else []
            # Match presence is separate from score: BM25 can produce zero/negative IDF.
            matching = [i for i, words in enumerate(self.doc_terms) if terms.intersection(words)]
            matching.sort(key=lambda i: (-float(scores[i]), chunk_id(self.all_docs[i])))
            bm25_results = [Document(page_content=self.all_docs[i].page_content,
                metadata={**self.all_docs[i].metadata, 'bm25_score': float(scores[i])})
                for i in matching[:candidate_k]]
        with timed('rrf_fusion'):
            fused = {}
            lists = [vector_results, bm25_results]
            if extra_docs:
                lists.append(extra_docs)
            for documents in lists:
                for rank, doc in enumerate(unique_docs(documents), 1):
                    identity = chunk_id(doc)
                    item = fused.setdefault(identity, {'score': 0.0, 'doc': doc})
                    item['score'] += 1.0 / (self.rrf_k + rank)
                    item['doc'].metadata.update(doc.metadata)
            ranked = sorted(fused.values(), key=lambda item: (-item['score'], chunk_id(item['doc'])))
            candidates = []
            for item in ranked[:self.rerank_k]:
                doc = item['doc']
                doc.metadata['rrf_score'] = item['score']
                candidates.append(doc)
        return self.rank_and_filter(query, candidates, top_k)

    def rank_and_filter(self, query, docs, top_k=5):
        docs = unique_docs(docs)
        if not docs:
            return []
        if self.reranker_model is not None:
            try:
                with timed('reranker'):
                    scores = self._rerank_scores(query, docs)
                scored = []
                for doc, score in zip(docs, scores):
                    doc.metadata.update(rerank_score=float(score), relevance_score=float(score),
                                        relevance_method='reranker')
                    if score >= self.min_rerank_score:
                        scored.append(doc)
                return sorted(scored, key=lambda d: d.metadata['rerank_score'], reverse=True)[:top_k]
            except RequestTimeout:
                raise
            except Exception:
                logger.exception('重排序失败，使用词项匹配过滤')
        selected = []
        for doc in docs:
            score = lexical_relevance(query, doc)
            doc.metadata.update(relevance_score=score, relevance_method='lexical_fallback')
            if score >= self.min_lexical_score and score > 0:
                selected.append(doc)
        return selected[:top_k]

    def _rerank_scores(self, query, docs):
        import torch
        resource = self.reranker_model
        batch_size = resource.get('batch_size', 4)
        device = resource.get('device', 'cpu')
        scores = []
        # Cached models are shared across sessions; avoid concurrent CPU/GPU oversubscription.
        with resource['lock']:
            for start in range(0, len(docs), batch_size):
                check_budget()
                pairs = [[query, doc.page_content] for doc in docs[start:start + batch_size]]
                inputs = resource['tokenizer'](pairs, padding=True, truncation=True,
                    max_length=resource.get('max_length', 512), return_tensors='pt')
                inputs = {key: value.to(device) for key, value in inputs.items()}
                with torch.inference_mode():
                    logits = resource['model'](**inputs).logits.reshape(-1).float()
                    scores.extend(torch.sigmoid(logits).cpu().tolist())
        return scores
