"""Aggregate saved blind AI judgments and real API telemetry. No model calls."""
from pathlib import Path
import json,csv,re,argparse,statistics
import numpy as np
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines()]
def avg(xs):return statistics.mean(xs) if xs else None
def ratio(a,b):return a/b if b else None
def dump(p,data):
    with p.open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2)
def csvwrite(p,rs):
    with p.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',default='paid_test_01');a=parser.parse_args()
    r=ROOT/'runs'/a.run_id; answers=lines(r/'answers.jsonl');calls=lines(r/'calls.jsonl')
    blind={x['answer_id']:x for x in read(r/'blind_answers.json')}
    reviews=read(r/'ai_judgments.json'); assert len(reviews)==len(blind)
    scores={x['answer_id']:x for x in reviews};assert len(scores)==len(blind) and set(scores)==set(blind)
    mapping={(x['question_id'],x['arm']):x['answer_id'] for x in read(r/'blind_mapping.json')}
    cases={c['question_id']:c for c in read(ROOT/'cases.json')};facts=read(ROOT/'required_facts.json')
    per=[]
    for x in answers:
        q=x['question_id'];arm=x['arm'];key=mapping[q,arm];s=scores[key];b=blind[key];state=x.get('state',{})
        assert s['correctness'] in (0,0.5,1)
        assert len(s['covered_fact_indices'])==len(set(s['covered_fact_indices']))
        assert all(1<=i<=len(facts[q]) for i in s['covered_fact_indices'])
        for claim in s['claims']:assert claim['verdict'] in ('supported','contradicted','insufficient')
        cs=[c for c in calls if c.get('question_id')==q and c.get('arm')==arm]
        if arm=='C':cs=cs+[c for c in calls if c.get('question_id')==q and c.get('arm')=='B']
        citations=[int(i) for i in re.findall(r'\[(\d+)\]',state.get('answer',''))]
        valid=sum(1<=i<=len(state.get('all_docs',[])) for i in citations)
        count={v:sum(c['verdict']==v for c in s['claims']) for v in ['supported','contradicted','insufficient']}
        cit_claims=[c for c in s['claims'] if c['citations']]
        answer_fail=bool(x.get('error') or state.get('generation_error') or x['status']!='completed')
        workflow_fail=answer_fail or (arm=='C' and state.get('verification_status')=='error')
        sec=x.get('generation_verification_seconds',0)+(x.get('B_generation_verification_seconds',0) if arm=='C' else 0)
        row={'question_id':q,'question_type':cases[q]['question_type'],'origin':cases[q]['origin'],'arm':arm,'answer_id':key,
          'answerable':cases[q]['answerable'],'correctness':s['correctness'],'fully_correct':int(s['correctness']==1),
          'availability_adjusted_correctness':0 if workflow_fail else s['correctness'],
          'reference_facts':len(facts[q]),'covered_facts':len(s['covered_fact_indices']),
          'completeness':ratio(len(s['covered_fact_indices']),len(facts[q])),
          'refused':s['refused'],'conflict_handled':s.get('conflict_handled'),
          'factual_claims':len(s['claims']),**count,'claims_with_citations':len(cit_claims),
          'citation_supported_claims':sum(bool(c['citation_support']) for c in cit_claims),
          'citation_occurrences':len(citations),'valid_citation_occurrences':valid,
          'answer_failure':answer_fail,'workflow_failure':workflow_fail,
          'verification_status':state.get('verification_status'),'rewrite_count':state.get('retry_count',0),
          'rewrite_attempted':any(c.get('arm')=='C' and c['messages'][0]['content'].startswith('修正回答中') for c in cs),
          'api_calls':len(cs),'api_failures':sum(bool(c.get('error')) for c in cs),
          'input_tokens_known':sum(c.get('input_tokens') or 0 for c in cs),'output_tokens_known':sum(c.get('output_tokens') or 0 for c in cs),
          'calls_missing_usage':sum(c.get('input_tokens') is None for c in cs),
          'fixed_evidence_pipeline_seconds':sec,'composed_with_retrieval_seconds':sec+x.get('retrieval_seconds_archived',0),
          'answer':state.get('answer',''),'review_notes':s['notes']}
        per.append(row)
    csvwrite(r/'answer_scores.csv',per);dump(r/'answer_scores.json',per)
    def aggregate(rs):
        pos=[x for x in rs if x['answerable']];neg=[x for x in rs if not x['answerable']];conf=[x for x in rs if x['question_type']=='conflict']
        successful=[x for x in rs if not x['workflow_failure']]
        total=lambda k:sum(x[k] for x in rs)
        return {'n':len(rs),'fully_correct':total('fully_correct'),'fully_correct_rate':avg([x['fully_correct'] for x in rs]),
          'correctness_score':avg([x['correctness'] for x in rs]),'availability_adjusted_score':avg([x['availability_adjusted_correctness'] for x in rs]),
          'successful_workflows':len(successful),'success_only_score':avg([x['correctness'] for x in successful]),
          'answer_failures':total('answer_failure'),'workflow_failures':total('workflow_failure'),
          'completeness_macro':avg([x['completeness'] for x in pos]),'factual_claims':total('factual_claims'),
          'supported':total('supported'),'contradicted':total('contradicted'),'insufficient':total('insufficient'),
          'supported_rate':ratio(total('supported'),total('factual_claims')),'contradicted_rate':ratio(total('contradicted'),total('factual_claims')),
          'insufficient_rate':ratio(total('insufficient'),total('factual_claims')),
          'citation_presence_rate':ratio(total('claims_with_citations'),total('factual_claims')),
          'citation_existence_rate':ratio(total('valid_citation_occurrences'),total('citation_occurrences')),
          'citation_support_rate':ratio(total('citation_supported_claims'),total('claims_with_citations')),
          'correct_refusal_rate':avg([int(x['refused'] and x['fully_correct']) for x in neg]),
          'false_refusal_rate':avg([int(x['refused']) for x in pos]),
          'conflict_correct_rate':avg([int(x['conflict_handled']) for x in conf]),
          'fixed_evidence_p50_seconds':float(np.percentile([x['fixed_evidence_pipeline_seconds'] for x in rs],50)),
          'fixed_evidence_p95_seconds':float(np.percentile([x['fixed_evidence_pipeline_seconds'] for x in rs],95)),
          'composed_p50_seconds':float(np.percentile([x['composed_with_retrieval_seconds'] for x in rs],50)),
          'composed_p95_seconds':float(np.percentile([x['composed_with_retrieval_seconds'] for x in rs],95)),
          'success_fixed_evidence_p50_seconds':float(np.percentile([x['fixed_evidence_pipeline_seconds'] for x in successful],50)) if successful else None,
          'api_calls_including_B_for_C':total('api_calls'),'input_tokens_known':total('input_tokens_known'),'output_tokens_known':total('output_tokens_known'),
          'calls_missing_usage':total('calls_missing_usage'),'rewrite_questions':total('rewrite_attempted')}
    summary=[]
    groups={'overall':lambda x:True,'repository_natural':lambda x:x['origin']=='repository_natural','synthetic':lambda x:x['origin'].startswith('synthetic')}
    groups.update({t:lambda x,t=t:x['question_type']==t for t in {x['question_type'] for x in per}})
    for group,select in groups.items():
        for arm in ['A','B','C']:summary.append({'group':group,'arm':arm,**aggregate([x for x in per if x['arm']==arm and select(x)])})
    dump(r/'answer_summary.json',summary);csvwrite(r/'answer_summary.csv',summary)
    rng=np.random.default_rng(20260925); paired=[]
    for first,second in [('B','A'),('C','B')]:
        for metric in ['fully_correct','correctness','availability_adjusted_correctness','completeness','fixed_evidence_pipeline_seconds']:
            ds=[]
            for q in sorted(cases):
                one=next(x for x in per if x['question_id']==q and x['arm']==first)[metric]
                two=next(x for x in per if x['question_id']==q and x['arm']==second)[metric]
                if one is not None and two is not None:ds.append(one-two)
            vals=np.array(ds);boot=vals[rng.integers(0,len(vals),(10000,len(vals)))].mean(axis=1)
            paired.append({'comparison':first+'-'+second,'metric':metric,'n':len(vals),'difference':float(vals.mean()),
               'ci_low':float(np.percentile(boot,2.5)),'ci_high':float(np.percentile(boot,97.5))})
    dump(r/'answer_paired.json',paired);csvwrite(r/'answer_paired.csv',paired)
    changes=[]
    for q in sorted(cases):
        b=next(x for x in per if x['question_id']==q and x['arm']=='B');c=next(x for x in per if x['question_id']==q and x['arm']=='C')
        changes.append({'question_id':q,'B':b['correctness'],'C':c['correctness'],'delta':c['correctness']-b['correctness'],
            'rewritten':bool(c['rewrite_count']),'rewrite_attempted':c['rewrite_attempted'],
            'C_workflow_failure':c['workflow_failure'],'before':b['answer'],'after':c['answer']})
    dump(r/'correction_changes.json',changes);csvwrite(r/'correction_changes.csv',changes)
    labels=['支持','矛盾','证据不足'];matrix={label:{x:0 for x in labels+['missing_or_error']} for label in labels}
    checker=lines(r/'checker_challenge.jsonl')
    for x in checker:
        checks=x.get('checks',[])
        pred=checks[0]['verdict'] if x['status']=='completed' and len(checks)==1 else 'missing_or_error'
        matrix[x['reference_verdict']][pred]+=1
    cls=[]
    for label in labels:
        tp=matrix[label][label];rec=ratio(tp,sum(matrix[label].values()));prec=ratio(tp,sum(matrix[k][label] for k in labels))
        cls.append({'label':label,'precision':prec,'recall':rec,'f1':2*prec*rec/(prec+rec) if prec is not None and prec+rec else 0})
    dump(r/'checker_confusion.json',{'matrix':matrix,'per_class':cls,'n':len(checker),
        'accuracy':sum(matrix[k][k] for k in labels)/len(checker),'scope':'30 synthetic atomic claims; not natural-response coverage'})
    print(json.dumps([x for x in summary if x['group']=='overall'],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
