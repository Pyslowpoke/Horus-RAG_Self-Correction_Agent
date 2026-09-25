"""Scripted contract probes, NOT real model accuracy measurements."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT.parents[1]))
from src.verifiers.fact_checker import FactChecker
from src.agents.fact_check_agent import make_fact_check_agent
from langchain_core.documents import Document

class Scripted:
    def __init__(self,reply): self.reply=reply; self.calls=0
    def generate(self,*a,**k): self.calls+=1; return self.reply

def main():
    ctx='住宿上限为600元。申请须提前3天。'
    examples=[
      ('valid_support','住宿上限为600元。',[{'claim':'住宿上限为600元','verdict':'支持','evidence':'住宿上限为600元'}]),
      ('real_quote_wrong_entailment','住宿上限为900元。',[{'claim':'住宿上限为900元','verdict':'支持','evidence':'住宿上限为600元'}]),
      ('omitted_false_claim','住宿上限为600元。申请须提前1天。',[{'claim':'住宿上限为600元','verdict':'支持','evidence':'住宿上限为600元'}]),
      ('empty_claims_for_factual_answer','住宿上限为900元。',[]),
      ('fabricated_quote','住宿上限为900元。',[{'claim':'住宿上限为900元','verdict':'支持','evidence':'住宿上限为900元'}]),
      ('invalid_json','住宿上限为900元。','not-json')]
    results=[]
    for name,answer,reply in examples:
        llm=Scripted(reply if isinstance(reply,str) else json.dumps(reply,ensure_ascii=False))
        state={'query':'住宿上限和申请提前时间是什么？','answer':answer,'context':ctx,
               'all_docs':[Document(page_content=ctx)],'search_mode':'local'}
        out=make_fact_check_agent(FactChecker(llm))(state)
        results.append({'probe':name,'context':ctx,'answer':answer,'scripted_checker_output':reply,
                        'actual_product_result':out,'fake_llm_calls':llm.calls,'paid_api_calls':0,
                        'interpretation':'Contract/adversarial stub probe; does not measure real LLM error frequency'})
    with (ROOT/'checker_contract_probes.json').open('x',encoding='utf-8') as f: json.dump(results,f,ensure_ascii=False,indent=2)
    print([(r['probe'],r['actual_product_result']['verification_status']) for r in results])

if __name__=='__main__': main()
