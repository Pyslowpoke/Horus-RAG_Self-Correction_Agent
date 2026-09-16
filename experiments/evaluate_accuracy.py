"""Offline retrieval evaluation using annotated evidence IDs or snippets.
Example: python -m experiments.evaluate_accuracy --cases experiments/retrieval_cases.example.json --output evaluation.json
No LLM calls. Source snippets are matched against the actual returned chunks.
"""
import argparse
import json
from pathlib import Path
import statistics
import time
from src.config import load_config
from src.components import load_embeddings, load_db, build_bm25, load_reranker, make_retriever
from src.documents import chunk_id
from src.runtime import RequestBudget, request_scope


def matching_ids(case, docs):
    if 'relevant_ids' in case:
        return set(case['relevant_ids'])
    snippets = case.get('evidence_contains', [])
    return {chunk_id(doc) for doc in docs if any(snippet in doc.page_content for snippet in snippets)}


def evaluate(retriever, cases, all_docs, top_k=5):
    rows = []
    for case in cases:
        expected = matching_ids(case, all_docs)
        answerable = case.get('answerable', True)
        if answerable and not expected:
            raise ValueError(f"评测证据不在当前索引中: {case['query']}")
        if not answerable and expected:
            raise ValueError('无答案问题不能配置相关证据')
        with request_scope(RequestBudget(120)) as budget:
            started = time.perf_counter()
            docs = retriever.retrieve(case['query'], top_k=top_k)
            elapsed = time.perf_counter() - started
        hits = [chunk_id(doc) in expected for doc in docs]
        first = next((i for i, hit in enumerate(hits, 1) if hit), None)
        rows.append({'query': case['query'], 'answerable': answerable,
            'precision_at_k': sum(hits) / top_k,
            'returned_precision': sum(hits) / len(docs) if docs else 0,
            'recall_at_k': sum(hits) / len(expected) if expected else None,
            'reciprocal_rank': 1 / first if first else 0,
            'correct_abstention': not docs if not answerable else None,
            'seconds': elapsed, 'returned': len(docs), 'timings': budget.metrics['timings'],
            'results': [{'chunk_id': chunk_id(d), 'source': d.metadata.get('source'),
                         'score': d.metadata.get('relevance_score')} for d in docs]})
    import numpy as np
    answerable_rows = [r for r in rows if r['answerable']]
    negative_rows = [r for r in rows if not r['answerable']]
    average = lambda values: statistics.mean(values) if values else None
    return {'summary': {
        'count': len(rows),
        'precision_at_k_answerable': average([r['precision_at_k'] for r in answerable_rows]),
        'recall_at_k': average([r['recall_at_k'] for r in answerable_rows]),
        'returned_precision_answerable': average([r['returned_precision'] for r in answerable_rows]),
        'mrr': average([r['reciprocal_rank'] for r in answerable_rows]),
        'correct_abstention_rate': average([r['correct_abstention'] for r in negative_rows]),
        'p50_seconds': float(np.percentile([r['seconds'] for r in rows], 50)),
        'p95_seconds': float(np.percentile([r['seconds'] for r in rows], 95))}, 'cases': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', required=True)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--output', default='retrieval_evaluation.json')
    parser.add_argument('--no-reranker', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    start = time.perf_counter()
    embeddings = load_embeddings(config)
    db = load_db(config, embeddings)
    bm25, docs = build_bm25(db)
    reranker = None if args.no_reranker else load_reranker(config)
    retriever = make_retriever(config, db, bm25, docs, reranker)
    initialization = time.perf_counter() - start
    # Warm-up is reported separately from the measured queries.
    start = time.perf_counter()
    retriever.retrieve('检索', top_k=config['retrieval']['top_k'])
    warmup = time.perf_counter() - start
    cases = json.loads(Path(args.cases).read_text(encoding='utf-8-sig'))
    if not cases:
        raise ValueError('评测集不能为空')
    result = evaluate(retriever, cases, docs, config['retrieval']['top_k'])
    result.update(initialization_seconds=initialization, warmup_seconds=warmup,
                  reranker_active=reranker is not None, embedding_model=config['embedding']['model_name'])
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({**result['summary'], 'initialization_seconds': initialization,
                      'reranker_active': result['reranker_active']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
