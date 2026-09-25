"""Run fixed offline A/B/V retrieval; never invokes LLMs or reads credentials."""
from pathlib import Path
import sys, json, time, argparse, random, hashlib, copy, os
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]))
os.environ['ANONYMIZED_TELEMETRY']='False'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import yaml
from langchain_core.documents import Document
from src.config import load_config
from src.components import load_embeddings, load_reranker, build_bm25, make_retriever
from src.data_ingestion import split_documents, sync_chunks, write_manifest
from src.documents import chunk_id
from src.runtime import RequestBudget, request_scope

def write(path,obj):
    with path.open('x',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False,indent=2)

def serialize(d): return {'chunk_id':chunk_id(d),'text':d.page_content,'metadata':d.metadata}
def norm(t): return ''.join(t.split())

def score(case,docs,error):
    refs=case['evidence']
    matrix=[[e['source']==d.metadata.get('source') and norm(e['quote']) in norm(d.page_content) for e in refs] for d in docs]
    hits=[any(row) for row in matrix]
    coverage=[any(row[i] for row in matrix) for i in range(len(refs))]
    good=not error
    return {'any_hit':int(good and any(hits)) if refs else None,
        'all_hit':int(good and all(coverage)) if refs else None,
        'evidence_recall':sum(coverage)/len(refs) if refs and good else (0 if refs else None),
        'precision_at_5':sum(hits)/5 if good else 0,
        'returned_precision':sum(hits)/len(docs) if docs and good else 0,
        'mrr':next((1/i for i,h in enumerate(hits,1) if h),0) if refs and good else (0 if refs else None),
        'empty_success':int(good and not docs),'evidence_covered':coverage}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run-id',required=True); p.add_argument('--repeats',type=int,default=3)
    args=p.parse_args(); out=ROOT/'runs'/args.run_id; out.mkdir(parents=True,exist_ok=False)
    # Refuse changed or stale dataset before initialization.
    for name,digest in json.loads((ROOT/'data_manifest.json').read_text(encoding='utf-8')).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, name
    cfg=load_config(ROOT/'config.original.yaml'); cfg=copy.deepcopy(cfg)
    cfg['db']={'docs_dir':str(ROOT/'corpus'),'chroma_persist_dir':str(out/'index')}
    cfg['generation']['streaming']=False
    write(out/'config.json',cfg)
    start=time.perf_counter(); emb=load_embeddings(cfg)
    documents=[Document(page_content=f.read_text(encoding='utf-8'),metadata={'source':f.name}) for f in sorted((ROOT/'corpus').glob('*.txt'))]
    chunks=split_documents(documents,emb,cfg)
    from langchain_community.vectorstores import Chroma
    db=Chroma(persist_directory=str(out/'index'),embedding_function=emb)
    ingest=sync_chunks(db,chunks); write_manifest(out/'index',cfg,len(chunks))
    bm25,docs=build_bm25(db); rr=load_reranker(cfg)
    assert rr is not None, 'Required reranker unavailable; no silent fallback for formal experiment'
    hybrid=make_retriever(cfg,db,bm25,docs,rr)
    init=time.perf_counter()-start
    write(out/'chunks.json',[serialize(d) for d in docs])
    cases=json.loads((ROOT/'cases.json').read_text(encoding='utf-8'))
    dev=json.loads((ROOT/'dev_cases.json').read_text(encoding='utf-8'))
    validation=[]
    for c in cases+dev:
        for e in c['evidence']:
            ids=[chunk_id(d) for d in docs if d.metadata['source']==e['source'] and norm(e['quote']) in norm(d.page_content)]
            validation.append({'question_id':c['question_id'],'quote':e['quote'],'chunk_ids':ids})
    write(out/'evidence_validation.json',validation)
    missing=[r for r in validation if not r['chunk_ids']]
    if missing: raise ValueError(f'{len(missing)} references cross chunks or are missing; inspect validation before formal run')
    import torch
    from huggingface_hub.constants import HF_HUB_CACHE
    cached={}
    for model in [cfg['embedding']['model_name'],cfg['reranker']['model_name']]:
        refs=Path(HF_HUB_CACHE)/('models--'+model.replace('/','--'))/'refs'
        cached[model]={r.name:r.read_text().strip() for r in refs.glob('*') if r.is_file()}
    write(out/'initialization.json',{'seconds':init,'ingestion':ingest,'device':rr['device'],
        'torch_threads':torch.get_num_threads(),'model_cached_refs':cached,'repeats':args.repeats,
        'api_calls':0,'result_cache':False,'question_order_seed':20260925})
    capture=[]
    original=hybrid.rank_and_filter
    def recording_rank(q,ds,k=5):
        capture.extend(serialize(d) for d in ds)
        return original(q,ds,k)
    hybrid.rank_and_filter=recording_rank
    def retrieve(arm,q):
        if arm=='B': return hybrid.retrieve(q,5)
        pairs=db.similarity_search_with_score(q,k=5 if arm=='A' else 10)
        result=[Document(page_content=d.page_content,metadata={**d.metadata,'vector_distance':float(s)}) for d,s in pairs]
        return result if arm=='A' else hybrid.rank_and_filter(q,result,5)
    def run(c,arm,repeat,f):
        capture.clear(); ds=[]; error=None
        budget=RequestBudget(120); started=time.perf_counter()
        try:
            with request_scope(budget): ds=retrieve(arm,c['question'])
        except Exception as exc: error=type(exc).__name__ # no exception payloads / secrets
        elapsed=time.perf_counter()-started
        fallback=any(d.metadata.get('relevance_method')=='lexical_fallback' for d in ds)
        row={'question_id':c['question_id'],'question_type':c['question_type'],'origin':c['origin'],
          'question':c['question'],'answerable':c['answerable'],'split':c['split'],'arm':arm,'repeat':repeat,
          'seconds':elapsed,'error':error,'fallback':fallback,'returned':len(ds),'metrics':budget.metrics,
          'scores':score(c,ds,error),'documents':[serialize(d) for d in ds],'candidates':list(capture)}
        f.write(json.dumps(row,ensure_ascii=False)+'\n'); f.flush()
        return row
    with (out/'dev.jsonl').open('x',encoding='utf-8') as f:
        for c in dev:
            for arm in ['A','B','V']: run(c,arm,0,f)
    print(f'DEV_COMPLETE index_chunks={len(docs)} initialization_seconds={init:.3f}',flush=True)
    rng=random.Random(20260925)
    with (out/'retrieval.jsonl').open('x',encoding='utf-8') as f:
        for rep in range(args.repeats):
            order=list(cases); rng.shuffle(order)
            for i,c in enumerate(order):
                arms=['A','B','V']; rng.shuffle(arms)
                for arm in arms: run(c,arm,rep,f)
                if (i+1)%10==0: print(f'repeat={rep+1} questions={i+1}/{len(cases)}',flush=True)
    print('COMPLETE: raw retrieval records saved; no LLM calls.',flush=True)

if __name__=='__main__': main()
