"""Post-hoc local scoring of saved rejected candidates; never replaces formal results."""
from pathlib import Path
import sys,json,time,os
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parents[1]))
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
from src.components import load_reranker
from src.config import load_config
from src.retrievers.hybrid_retriever import HybridRetriever
from langchain_core.documents import Document
def main():
    run=ROOT/'runs/local_01'
    assert not (run/'rerank_posthoc.json').exists()
    start=time.perf_counter();rr=load_reranker(load_config(ROOT/'config.original.yaml'));assert rr is not None
    ranker=HybridRetriever(None,None,[],reranker_model=rr)
    rows=[json.loads(l) for l in (run/'retrieval.jsonl').read_text(encoding='utf-8').splitlines()]
    results=[]
    for r in rows:
        if r['arm']!='B' or r['repeat']!=0 or not r['answerable'] or r['scores']['all_hit']:continue
        docs=[Document(page_content=d['text'],metadata=d['metadata']) for d in r['candidates']]
        scores=ranker._rerank_scores(r['question'],docs)
        selected=[d.metadata['chunk_id'] for d,s in sorted(zip(docs,scores),key=lambda x:-x[1]) if s>=.3][:5]
        results.append({'question_id':r['question_id'],'formal_final_ids':[d['chunk_id'] for d in r['documents']],
           'posthoc_final_ids':selected,'same_final_ids':selected==[d['chunk_id'] for d in r['documents']],
           'candidates':[{'id':d.metadata['chunk_id'],'text':d.page_content,'source':d.metadata['source'],'score':s} for d,s in zip(docs,scores)]})
    (run/'rerank_posthoc.json').write_text(json.dumps({'scope':'Post-hoc diagnosis on saved candidates; no threshold changes, no replacement of formal metrics',
        'seconds_including_model_load':time.perf_counter()-start,'rows':results},ensure_ascii=False,indent=2),encoding='utf-8')
    print('Candidate audit complete:',len(results),'all selections match:',all(r['same_final_ids'] for r in results))
if __name__=='__main__':main()
