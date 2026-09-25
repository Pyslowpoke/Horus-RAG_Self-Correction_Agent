"""Post-hoc diagnosis on immutable raw records; never calls models or tunes settings."""
from pathlib import Path
import json,sys,collections
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parents[1]))
from src.verifiers.fact_checker import FactChecker
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines()]
def save(p,obj):
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
def main():
    run=ROOT/'runs/paid_test_01';calls=lines(run/'calls.jsonl');checker=FactChecker(None);out=[]
    for r in calls:
        prompt=r['messages'][0]['content']
        if not prompt.startswith('逐条核对'):continue
        row={k:r.get(k) for k in ['call','question_id','claim_id','arm','finish_reason','error']}
        if r.get('error'):row['parse_status']='api_error'
        else:
            try:
                entries=checker._parse_json_response(r['answer'])
                # The corpus includes prompt source code. Split at the LAST actual instructions,
                # not the first occurrence of the same words within a quoted document.
                context=prompt.split('参考资料：',1)[1].rsplit('\n返回 JSON 数组，每项包含 claim、verdict、evidence。',1)[0]
                valid=[checker._validate_log_entry(e,context) for e in entries]
                row.update(parse_status='valid',raw_entries=entries,validated_entries=valid,
                    quote_downgrades=sum(e.get('verdict')!=v['verdict'] for e,v in zip(entries,valid)))
            except Exception as exc:row.update(parse_status=str(exc),response_excerpt=r['answer'][:200])
        out.append(row)
    save(run/'checker_diagnostics_v2.json',out)
    cases={c['question_id']:c for c in read(ROOT/'cases.json')}
    rows=lines(ROOT/'runs/local_01/retrieval.jsonl')
    norm=lambda s:''.join(s.split())
    def covered(e,ds):return any(d['metadata'].get('source')==e['source'] and norm(e['quote']) in norm(d['text']) for d in ds)
    failure=[]
    for r in rows:
        if r['arm']!='B' or r['repeat']!=0 or not r['answerable'] or r['scores']['all_hit']:continue
        c=cases[r['question_id']]; ref=[]
        for e in c['evidence']:
            candidates=[d for d in r['candidates'] if covered(e,[d])]
            ref.append({'quote':e['quote'],'in_final':covered(e,r['documents']),
               'in_rerank_candidates':bool(candidates),'candidate_scores':[d['metadata'].get('rerank_score') for d in candidates]})
        failure.append({'question_id':r['question_id'],'question':r['question'],'returned':r['returned'],'reference_trace':ref,
                        'limitation':'Exact annotated source quotes only; equivalent evidence elsewhere may be unlabelled.'})
    save(ROOT/'runs/local_01/failure_diagnosis.json',failure)
    tokens={};all_calls=[]
    for name in ['paid_dev_01','paid_test_01']:
        rs=lines(ROOT/'runs'/name/'calls.jsonl');all_calls+=rs
        tokens[name]={'calls':len(rs),'failed_calls':sum(bool(r['error']) for r in rs),
          'input_tokens_known':sum(r.get('input_tokens') or 0 for r in rs),
          'output_tokens_known':sum(r.get('output_tokens') or 0 for r in rs),
          'usage_missing_calls':sum(r.get('input_tokens') is None for r in rs),
          'returned_models':dict(collections.Counter(r.get('response_model','no_response') for r in rs)),
          'returned_fingerprints':dict(collections.Counter(str(r.get('system_fingerprint')) for r in rs))}
    tokens['total']={'calls':len(all_calls),'failed_calls':sum(bool(r['error']) for r in all_calls),
       'input_tokens_known':sum(r.get('input_tokens') or 0 for r in all_calls),'output_tokens_known':sum(r.get('output_tokens') or 0 for r in all_calls),
       'currency_cost':None,'retries':0,'usage_caveat':'Usage missing for timed-out calls; known totals are not a complete bill.'}
    save(ROOT/'api_usage_final.json',tokens)
    print(json.dumps({'usage':tokens,'B_incomplete_references':failure},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
