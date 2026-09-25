"""Export anonymized unique responses. Exact duplicates share one AI review."""
from pathlib import Path
import json,random,argparse
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',default='paid_test_01');args=p.parse_args()
    base=ROOT/'runs'/args.run_id
    answers=[json.loads(l) for l in (base/'answers.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(answers)==180, 'Wait for all formal rows, including failures'
    cases={c['question_id']:c for c in json.loads((ROOT/'cases.json').read_text(encoding='utf-8'))}
    groups={}; mapping=[]
    for r in answers:
        s=r.get('state',{}); key=(r['question_id'],s.get('answer',''),s.get('context',''))
        groups.setdefault(key,[]).append(r)
    order=list(groups);random.Random(972025).shuffle(order)
    records=[]
    for n,key in enumerate(order,1):
        rs=groups[key]; r=rs[0];c=cases[r['question_id']];s=r.get('state',{});rid=f'anon_{n:03}'
        records.append({'answer_id':rid,'question_id':c['question_id'],'question_type':c['question_type'],
            'question':c['question'],'answerable':c['answerable'],'reference_answer':c['reference_answer'],
            'reference_evidence':c['evidence'],'answer':s.get('answer',''),
            'numbered_context':s.get('context',''),'documents':s.get('all_docs',[]),
            'execution_failed':bool(r.get('error') or s.get('generation_error') or r['status']!='completed')})
        mapping.extend({'answer_id':rid,'question_id':x['question_id'],'arm':x['arm']} for x in rs)
    for name,data in [('blind_answers.json',records),('blind_mapping.json',mapping)]:
        with (base/name).open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2)
    print('Unique anonymous responses:',len(records),'original rows:',len(answers))
if __name__=='__main__':main()
