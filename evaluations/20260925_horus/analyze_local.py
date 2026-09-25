"""Summaries and paired question-level bootstrap, no LLM scoring."""
from pathlib import Path
import argparse,json,csv,statistics
import numpy as np
ROOT=Path(__file__).resolve().parent
METRICS=['any_hit','all_hit','evidence_recall','precision_at_5','returned_precision','mrr']
def mean(xs): return statistics.mean(xs) if xs else None
def write_csv(path,rows):
    with path.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def dump(path,obj):
    with path.open('x',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False,indent=2)
def summarize(rs):
    pos=[r for r in rs if r['answerable']]; neg=[r for r in rs if not r['answerable']]
    result={'questions':len(set(r['question_id'] for r in rs)),'requests':len(rs),'failures':sum(bool(r['error']) for r in rs),
        'fallbacks':sum(r['fallback'] for r in rs),'positive_questions':len(set(r['question_id'] for r in pos)),
        'negative_questions':len(set(r['question_id'] for r in neg))}
    result.update({k:mean([r['scores'][k] for r in pos]) for k in METRICS})
    result.update(negative_empty_rate=mean([r['scores']['empty_success'] for r in neg]),
        positive_empty_rate=mean([r['scores']['empty_success'] for r in pos]),
        p50_seconds=float(np.percentile([r['seconds'] for r in rs],50)),
        p95_seconds=float(np.percentile([r['seconds'] for r in rs],95)),
        mean_seconds=mean([r['seconds'] for r in rs]),
        success_p50_seconds=float(np.percentile([r['seconds'] for r in rs if not r['error']],50)) if any(not r['error'] for r in rs) else None,
        api_calls=sum(r['metrics']['llm_calls'] for r in rs))
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',default='local_01');args=p.parse_args();root=ROOT/'runs'/args.run_id
    rows=[json.loads(l) for l in (root/'retrieval.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==540, f'Expected 60 x 3 arms x 3 repetitions, got {len(rows)}; no partial sample deletion'
    assert len({(r['question_id'],r['arm'],r['repeat']) for r in rows})==540
    summary=[]
    groups={'overall':lambda r:True,'repository_natural':lambda r:r['origin']=='repository_natural',
            'synthetic':lambda r:r['origin'].startswith('synthetic')}
    groups.update({t:(lambda r,t=t:r['question_type']==t) for t in sorted({r['question_type'] for r in rows})})
    for group,select in groups.items():
        for arm in ['A','B','V']:
            summary.append({'group':group,'arm':arm,**summarize([r for r in rows if r['arm']==arm and select(r)])})
    write_csv(root/'summary.csv',summary); dump(root/'summary.json',summary)
    per_case=[]
    for q in sorted({r['question_id'] for r in rows}):
        for arm in ['A','B','V']:
            rs=[r for r in rows if r['question_id']==q and r['arm']==arm]
            row={'question_id':q,'arm':arm,'type':rs[0]['question_type'],'origin':rs[0]['origin'],**summarize(rs)}
            row['stable_document_ids']=len({tuple(d['chunk_id'] for d in r['documents']) for r in rs})==1
            row['stable_scores']=len({json.dumps(r['scores'],sort_keys=True) for r in rs})==1
            per_case.append(row)
    write_csv(root/'per_question.csv',per_case)
    rng=np.random.default_rng(20260925); comparisons=[]
    for first,second in [('B','A'),('B','V')]:
        for metric in ['any_hit','all_hit','evidence_recall','mrr','mean_seconds','negative_empty_rate','positive_empty_rate']:
            ds=[]
            for q in sorted({r['question_id'] for r in rows}):
                a=next(r for r in per_case if r['question_id']==q and r['arm']==first)[metric]
                b=next(r for r in per_case if r['question_id']==q and r['arm']==second)[metric]
                if a is not None and b is not None: ds.append(a-b)
            values=np.array(ds); boot=values[rng.integers(0,len(values),(10000,len(values)))].mean(axis=1)
            comparisons.append({'comparison':f'{first}-{second}','metric':metric,'paired_questions':len(ds),
                'difference':float(values.mean()),'ci_low':float(np.percentile(boot,2.5)),
                'ci_high':float(np.percentile(boot,97.5)),'improved':int((values>0).sum()),
                'worsened':int((values<0).sum()),'unchanged':int((values==0).sum())})
    write_csv(root/'paired_bootstrap.csv',comparisons); dump(root/'paired_bootstrap.json',comparisons)
    repeats=[{'repeat':rep,'arm':arm,**summarize([r for r in rows if r['repeat']==rep and r['arm']==arm])}
        for rep in range(3) for arm in ['A','B','V']]
    write_csv(root/'repetitions.csv',repeats)
    # All questions are reviewable, not only favorable cases. Human fields remain empty.
    review=[]
    for c in json.loads((ROOT/'cases.json').read_text(encoding='utf-8')):
        a=next(r for r in per_case if r['question_id']==c['question_id'] and r['arm']=='A')
        b=next(r for r in per_case if r['question_id']==c['question_id'] and r['arm']=='B')
        review.append({'question_id':c['question_id'],'type':c['question_type'],'origin':c['origin'],'question':c['question'],
            'reference_answer':c['reference_answer'],'A_all_hit':a['all_hit'],'B_all_hit':b['all_hit'],
            'B_empty':b['positive_empty_rate'] if c['answerable'] else b['negative_empty_rate'],
            'evidence_json':json.dumps(c['evidence'],ensure_ascii=False),'answer_status':'not_generated',
            'review_status':'待人工复核','human_reference_valid':'','human_answer_grade':'','human_notes':'','reviewer':'','reviewed_at':''})
    write_csv(root/'human_review.csv',review)
    # Explicit missing values for requested answer-level metrics: not zero and not retrieval surrogates.
    dump(root/'answer_metrics_status.json',{'A':'not_run_by_offline_entry','B':'not_run_by_offline_entry',
      'C':'not_run_by_offline_entry','answer_correctness':None,'faithfulness_supported':None,
      'faithfulness_contradicted':None,'faithfulness_insufficient':None,'completeness':None,
      'citation_semantic_support':None,'answer_refusal_rate':None,'checker_confusion_matrix':None,
      'correction_wrong_to_right':None,'correction_right_to_wrong':None,
      'api_calls_this_evaluation':0,'api_input_tokens':0,'api_output_tokens':0,
      'warning':'Local retrieval and scripted probes cannot establish answer-quality improvement.'})
    print(json.dumps([r for r in summary if r['group']=='overall'],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
