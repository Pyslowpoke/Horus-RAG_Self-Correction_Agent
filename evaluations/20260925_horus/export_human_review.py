"""Export pending human review; never fills human judgments."""
from pathlib import Path
import json,csv,random
ROOT=Path(__file__).resolve().parent
def main():
    run=ROOT/'runs/paid_test_01';rs=json.loads((run/'answer_scores.json').read_text(encoding='utf-8'))
    cases={c['question_id']:c for c in json.loads((ROOT/'cases.json').read_text(encoding='utf-8'))}
    answers={(r['question_id'],r['arm']):r for r in [json.loads(l) for l in (run/'answers.jsonl').read_text(encoding='utf-8').splitlines()]}
    rows=[]
    for r in rs:
        c=cases[r['question_id']];state=answers[r['question_id'],r['arm']].get('state',{})
        strata=[r['question_type'],'correct' if r['correctness']==1 else 'partial_or_wrong']
        if r['refused']:strata.append('refusal')
        if r['rewrite_attempted']:strata.append('rewrite')
        if r['workflow_failure']:strata.append('workflow_failure')
        rows.append({'question_id':r['question_id'],'arm':r['arm'],'answer_id':r['answer_id'],'strata':'|'.join(strata),
            'question':c['question'],'reference':c['reference_answer'],'answer':r['answer'],
            'actual_context':state.get('context',''),'reference_evidence':json.dumps(c['evidence'],ensure_ascii=False),
            'AI_score':r['correctness'],'AI_notes':r['review_notes'],'status':'待人工复核',
            'human_score':'','human_claim_labels':'','human_citation_support':'','human_notes':'','reviewer':'','reviewed_at':''})
    rng=random.Random(20260925);order=list(rows);rng.shuffle(order);sample=[];seen=set()
    for tag in ['single','integration','constraint','unanswerable','conflict','false_premise','correct','partial_or_wrong','refusal','rewrite','workflow_failure']:
        candidates=[r for r in order if tag in r['strata'].split('|') and (r['question_id'],r['arm']) not in seen]
        for r in candidates[:2]:sample.append(r);seen.add((r['question_id'],r['arm']))
    for name,data in [('human_review_answers.csv',rows),('human_review_sample.csv',sample)]:
        with (ROOT/name).open('x',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(data)
    print('All rows',len(rows),'stratified sample',len(sample),'human judgments filled',0)
if __name__=='__main__':main()
