"""Explicitly gated paid phase. Credentials use the process environment.

Optional explicitly requested local dotenv loading; never logs credentials.
No fallback providers, application memory, web, or result cache.
Replays frozen retrieval evidence; measures generation/checking wall time separately.
Full UI end-to-end latency is NOT measured by this controlled experiment.
"""
from pathlib import Path
import argparse,sys,json,time,os,random,hashlib
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT.parents[1]))
from langchain_core.documents import Document
from src.config import load_config
from src.graph.multi_agent_graph import build_multi_agent_rag_graph
from src.verifiers.fact_checker import FactChecker
from src.runtime import RequestBudget,request_scope,record_usage,bounded_timeout

class StopExperiment(RuntimeError): pass

class Ledger:
    def __init__(self,path,max_calls):
        self.file=path.open('x',encoding='utf-8'); self.calls=0; self.max_calls=max_calls
        self.consecutive_failures=0; self.context={}; self.inputs=0; self.outputs=0
    def append(self,row): self.file.write(json.dumps(row,ensure_ascii=False)+'\n'); self.file.flush()

class AuditedLLM:
    def __init__(self,opts,ledger,role):
        import openai
        key=os.environ.get(opts['api_key_env'])
        if not key: raise ValueError('Missing injected environment variable: '+opts['api_key_env'])
        self.client=openai.OpenAI(api_key=key,base_url=opts['api_base'],max_retries=0)
        self.opts=opts; self.ledger=ledger; self.role=role
    def generate(self,messages,**kwargs):
        l=self.ledger
        if l.calls>=l.max_calls or l.consecutive_failures>=3: raise StopExperiment('Budget or consecutive-failure stop')
        l.calls+=1; record_usage(); start=time.perf_counter()
        row={**l.context,'call':l.calls,'role':self.role,'requested_model':self.opts['model'],
             'messages':messages,'temperature':0.1,'max_tokens':512,'retry':0,'stream':False}
        try:
            response=self.client.chat.completions.create(model=self.opts['model'],messages=messages,
                temperature=0.1,max_tokens=512,timeout=bounded_timeout(self.opts.get('timeout',15)))
            record_usage(response); usage=response.usage
            row.update(response_model=response.model,system_fingerprint=getattr(response,'system_fingerprint',None),
                answer=response.choices[0].message.content or '',finish_reason=response.choices[0].finish_reason,
                input_tokens=usage.prompt_tokens if usage else None,output_tokens=usage.completion_tokens if usage else None,
                error=None)
            l.inputs+=row['input_tokens'] or 0; l.outputs+=row['output_tokens'] or 0; l.consecutive_failures=0
            return row['answer']
        except Exception as exc:
            l.consecutive_failures+=1
            row.update(error=type(exc).__name__,status_code=getattr(exc,'status_code',None),input_tokens=None,output_tokens=None)
            raise RuntimeError('API call failed; inspect sanitized audit status') from None
        finally:
            row['seconds']=time.perf_counter()-start; l.append(row)

class FixedRetriever:
    def __init__(self,row): self.docs=[Document(page_content=d['text'],metadata=d['metadata']) for d in row['documents']]
    def retrieve(self,*a,**k): return self.docs

class Replay:
    def __init__(self,answer): self.answer=answer
    def generate(self,*a,**k): return self.answer

def clean_state(state):
    return {k:([{'text':d.page_content,'metadata':d.metadata} for d in v] if k in ('all_docs','retrieved_docs','web_docs') else v)
            for k,v in state.items()}

def execute(case,row,heavy,light,verify):
    graph=build_multi_agent_rag_graph(heavy,light,FixedRetriever(row),None,FactChecker(light),None,
        system_identity='Horus',max_retries=1,verification_enabled=verify,streaming=False,
        context_max_chars=6000,top_k=5)
    budget=RequestBudget(60); start=time.perf_counter(); error=None; result={}
    try:
        with request_scope(budget):
            result=graph.invoke({'query':case['question'],'optimized_query':case['question'],
                'search_mode':'local','top_k':5,'retry_count':0,'enhanced_context':'','preferences':{}})
    except Exception as exc: error=type(exc).__name__
    return {'state':clean_state(result),'error':error,'generation_verification_seconds':time.perf_counter()-start,
            'metrics':budget.metrics,'retrieval_seconds_archived':row['seconds'],
            'latency_scope':'fixed-evidence graph replay; archived retrieval is separate, not full UI E2E'}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run-id',required=True); p.add_argument('--retrieval-run',default='local_01')
    p.add_argument('--split',choices=['dev','test'],required=True); p.add_argument('--max-calls',type=int,required=True)
    p.add_argument('--budget-authorized',action='store_true')
    p.add_argument('--load-project-env',action='store_true',help='User-requested local dotenv credential loading; no values logged')
    args=p.parse_args()
    if not args.budget_authorized: p.error('Paid calls require explicit budget authorization')
    if args.max_calls<1: p.error('max-calls must be positive')
    cfg=load_config(ROOT/'config.original.yaml')
    if args.load_project_env:
        from dotenv import load_dotenv
        load_dotenv(ROOT.parents[1]/'.env',override=False)
    # Check presence only; never include credential values in any exception/log.
    absent=[cfg['llm'][role]['api_key_env'] for role in ['primary','light'] if not os.environ.get(cfg['llm'][role]['api_key_env'])]
    if absent: p.error('Inject credentials into process environment locally: '+', '.join(absent))
    if args.split=='test' and not (ROOT/'PAID_PILOT_REVIEWED.json').exists():
        p.error('Review actual pilot and renewed formal budget first; create PAID_PILOT_REVIEWED.json with reviewed run ID and authorized cap')
    out=ROOT/'runs'/args.run_id; out.mkdir(parents=True,exist_ok=False)
    ledger=Ledger(out/'calls.jsonl',args.max_calls)
    heavy=AuditedLLM(cfg['llm']['primary'],ledger,'generation'); light=AuditedLLM(cfg['llm']['light'],ledger,'checker_or_rewrite')
    cases=json.loads((ROOT/('dev_cases.json' if args.split=='dev' else 'cases.json')).read_text(encoding='utf-8'))
    inputfile=ROOT/'runs'/args.retrieval_run/('dev.jsonl' if args.split=='dev' else 'retrieval.jsonl')
    source=[json.loads(l) for l in inputfile.read_text(encoding='utf-8').splitlines()]
    lookup={(r['question_id'],r['arm']):r for r in source if r['repeat']==0}
    rows=[]; rng=random.Random(20260925); order=list(cases); rng.shuffle(order)
    stopped=False
    with (out/'answers.jsonl').open('x',encoding='utf-8') as f:
        for c in order:
            baseline=None
            for arm in ['A','B','C']:
                row={'question_id':c['question_id'],'question_type':c['question_type'],'arm':arm,'split':args.split}
                ledger.context={'question_id':c['question_id'],'arm':arm}
                stop=stopped or ledger.calls>=ledger.max_calls or ledger.consecutive_failures>=3
                if stop:
                    stopped=True; row.update(status='not_run_budget_or_failure_stop')
                elif arm=='C' and (not baseline or baseline.get('error') or baseline['state'].get('generation_error')):
                    row.update(status='not_run_B_generation_failed')
                else:
                    retrieved=lookup[c['question_id'],'B' if arm=='C' else arm]
                    if retrieved['error']:
                        row.update(status='not_run_retrieval_failed')
                    else:
                        generator=Replay(baseline['state']['answer']) if arm=='C' else heavy
                        result=execute(c,retrieved,generator,light,arm=='C'); row.update(result,status='completed')
                        if arm=='B': baseline=result
                        if arm=='C':
                            assert result.get('error') or result['state'].get('context')==baseline['state'].get('context')
                            row['initial_answer']=baseline['state']['answer']
                            row['B_generation_verification_seconds']=baseline['generation_verification_seconds']
                            row['B_metrics']=baseline['metrics']
                f.write(json.dumps(row,ensure_ascii=False)+'\n'); f.flush(); rows.append(row)
    if args.split=='test':
        challenge=json.loads((ROOT/'checker_cases.json').read_text(encoding='utf-8'))
        with (out/'checker_challenge.jsonl').open('x',encoding='utf-8') as f:
            for c in challenge:
                row=dict(c); ledger.context={'claim_id':c['claim_id'],'arm':'checker_challenge'}
                if ledger.calls>=ledger.max_calls or ledger.consecutive_failures>=3: row['status']='not_run_stop'
                else:
                    try:
                        with request_scope(RequestBudget(60)):
                            row['checks']=FactChecker(light).check_once(c['claim'],c['context'])
                        row['status']='completed'
                    except Exception as exc: row.update(status='error',error=type(exc).__name__)
                f.write(json.dumps(row,ensure_ascii=False)+'\n'); f.flush()
    ledger.file.close()
    summary={'calls':ledger.calls,'input_tokens_known':ledger.inputs,'output_tokens_known':ledger.outputs,
       'max_calls':args.max_calls,'split':args.split,'stopped':stopped,'currency_cost':None,
       'scoring_status':'pending independent blind scoring and human review','full_UI_E2E_measured':False}
    (out/'paid_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
