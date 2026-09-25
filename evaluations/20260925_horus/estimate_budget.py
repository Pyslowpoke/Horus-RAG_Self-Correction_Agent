"""Estimate from actual development retrieval; surrogate tokens, not billable usage."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT.parents[1]))
from src.generators.prompt_templates import GENERATION_PROMPT,FACT_CHECK_PROMPT,REWRITE_RESPONSE_PROMPT
def context_of(row):
    parts=[]; remaining=6000
    for i,d in enumerate(row['documents'][:5],1):
        prefix=f"[{i}] 来源: {d['metadata'].get('source','未知')}\n"
        available=remaining-len(prefix)-2
        if available<50: break
        part=prefix+d['text'][:available]; parts.append(part); remaining-=len(part)+2
    return '\n\n'.join(parts) if parts else '未找到相关文档。'
def main():
    import tiktoken
    enc=tiktoken.get_encoding('cl100k_base')
    rows=[json.loads(l) for l in (ROOT/'runs/local_01/dev.jsonl').read_text(encoding='utf-8').splitlines()]
    data=[]
    for r in rows:
        if r['arm'] not in ['A','B']: continue
        ctx=context_of(r)
        gen=GENERATION_PROMPT.format(system_identity='Horus',context=ctx,enhanced_context='',query=r['question'])
        check=FACT_CHECK_PROMPT.format(response='',context=ctx)
        rewrite=REWRITE_RESPONSE_PROMPT.format(original_response='',failed_claims='[]',context=ctx)
        data.append({'question_id':r['question_id'],'arm':r['arm'],'returned':r['returned'],
           'generation_prompt_proxy_tokens':len(enc.encode(gen))+16,
           'check_template_proxy_tokens':len(enc.encode(check))+16,
           'rewrite_template_proxy_tokens':len(enc.encode(rewrite))+16})
    active=[d for d in data if d['returned']]
    gen_calls=len(active); bs=[d for d in active if d['arm']=='B']
    pilot_calls=gen_calls+3*len(bs)
    pilot_input=sum(d['generation_prompt_proxy_tokens'] for d in active)+sum(2*d['check_template_proxy_tokens']+d['rewrite_template_proxy_tokens']+4*512 for d in bs)
    out={'basis':'18 real local dev retrievals, A/B prompts; cl100k_base surrogate, not DeepSeek/Qwen tokenizer; no remote pilot yet',
      'dev_questions':6,'dev_actual_generation_eligible':gen_calls,'dev_B_eligible':len(bs),
      'dev_estimated_max_calls':pilot_calls,'dev_estimated_input_proxy_tokens':pilot_input,'dev_output_token_cap':512*pilot_calls,
      'formal_60_extrapolated_input_proxy_tokens':10*pilot_input,
      'formal_max_calls_without_judge':60*5+30,'formal_max_output_tokens_without_judge':(60*5+30)*512,
      'independent_judge_optional_max_calls':180,'independent_judge_optional_output_cap':180*1024,
      'currency_cost':None,'currency_reason':'No confirmed account billing rates; no paid call authorized',
      'input_limit_note':'Input estimate scales dev only, may differ from formal questions; checker challenge prompts additional. Not a hard spend cap.',
      'rows':data}
    with (ROOT/'budget_estimate.json').open('x',encoding='utf-8') as f: json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in out.items() if k!='rows'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
