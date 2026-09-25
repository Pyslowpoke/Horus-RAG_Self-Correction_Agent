"""Compact read-only grader input: deduplicate passage text within each batch."""
from pathlib import Path
import argparse,json,hashlib
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--start',type=int,default=0);p.add_argument('--count',type=int,default=15);a=p.parse_args()
    rs=json.loads((ROOT/'runs/paid_test_01/blind_answers.json').read_text(encoding='utf-8'))[a.start:a.start+a.count]
    facts=json.loads((ROOT/'required_facts.json').read_text(encoding='utf-8'))
    docs={}
    for r in rs:
        numbered={}
        for i,d in enumerate(r['documents'],1):
            key=hashlib.sha256((d['metadata'].get('source','')+d['text']).encode()).hexdigest()[:8]
            docs[key]={'source':d['metadata'].get('source'),'text':d['text']};numbered[i]=key
        print(json.dumps({'id':r['answer_id'],'q':r['question'],'qid':r['question_id'],'required_facts':facts[r['question_id']],
            'answer':r['answer'],'citation_map':numbered,'failed':r['execution_failed']},ensure_ascii=False))
    print('PASSAGE_TEXTS')
    print(json.dumps(docs,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
