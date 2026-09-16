"""Compare embeddings on identical passages, then compare configured hybrid pipelines."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.config import load_config
from src.components import load_embeddings, load_db, build_bm25, load_reranker, make_retriever
from experiments.evaluate_accuracy import evaluate


class FixedPassagesRetriever:
    def __init__(self, embedding, docs):
        self.embedding, self.docs = embedding, docs
        self.matrix = np.asarray(embedding.embed_documents([doc.page_content for doc in docs]))
    def retrieve(self, query, top_k=5):
        vector = np.asarray(self.embedding.embed_query(query))
        distances = np.sum((self.matrix - vector) ** 2, axis=1)
        return [self.docs[i] for i in np.argsort(distances)[:top_k]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-config', required=True)
    parser.add_argument('--candidate-config', required=True)
    parser.add_argument('--cases', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    base_config, new_config = load_config(args.baseline_config), load_config(args.candidate_config)
    base_embedding, new_embedding = load_embeddings(base_config), load_embeddings(new_config)
    base_db, new_db = load_db(base_config, base_embedding), load_db(new_config, new_embedding)
    base_bm25, base_docs = build_bm25(base_db)
    new_bm25, new_docs = build_bm25(new_db)
    cases = json.loads(Path(args.cases).read_text(encoding='utf-8-sig'))
    positives = [c for c in cases if c.get('answerable', True)]
    report = {'fixed_passage_count': len(base_docs), 'models': {}, 'vector': {}, 'hybrid': {}}
    # Fixed old passages remove chunking differences from the pure embedding comparison.
    for name, config, embedding in [('baseline', base_config, base_embedding), ('chinese', new_config, new_embedding)]:
        retriever = FixedPassagesRetriever(embedding, base_docs)
        retriever.retrieve('检索')
        result = evaluate(retriever, positives, base_docs, 5)
        report['vector'][name] = result
        report['models'][name] = config['embedding']
        print('vector', name, json.dumps(result['summary'],ensure_ascii=False),flush=True)
    reranker = load_reranker(new_config)
    if reranker is None:
        raise RuntimeError('重排器未加载，不能验证完整混合检索')
    for name, config, db, bm25, docs in [('baseline',base_config,base_db,base_bm25,base_docs),
                                      ('chinese',new_config,new_db,new_bm25,new_docs)]:
        retriever = make_retriever(config,db,bm25,docs,reranker)
        retriever.retrieve('检索')
        result = evaluate(retriever,cases,docs,config['retrieval']['top_k'])
        report['hybrid'][name] = result
        print('hybrid',name,json.dumps(result['summary'],ensure_ascii=False),flush=True)
    Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
