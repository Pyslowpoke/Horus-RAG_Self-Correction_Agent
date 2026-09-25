"""Create auditable public-project and explicitly synthetic evaluation data. No API calls."""
from pathlib import Path
import json, hashlib, shutil

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]

def save(name, obj):
    p = ROOT / name
    if p.exists():
        raise FileExistsError(p)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def main():
    assert not (ROOT / 'cases.json').exists(), 'Refusing to overwrite data'
    corpus = ROOT / 'corpus'
    corpus.mkdir(exist_ok=True)
    files = ['README.zh-CN.md', 'docs/optimization.md', 'config.yaml',
             'src/verifiers/fact_checker.py', 'src/agents/router_agent.py',
             'src/agents/memory_agent.py', 'src/graph/multi_agent_graph.py',
             'src/generators/prompt_templates.py']
    sources = {}
    for name in files:
        text = (PROJECT / name).read_text(encoding='utf-8-sig')
        target = name.replace('/', '__') + '.txt'
        (corpus / target).write_text(text, encoding='utf-8')
        sources[name] = {'corpus_file': target, 'text': text, 'origin': 'repository_natural',
                         'sha256': hashlib.sha256(text.encode()).hexdigest()}
    R, O = 'README.zh-CN.md', 'docs/optimization.md'
    cases = []
    def add(kind, q, answer, refs, origin='repository_natural', behavior='answer_with_evidence'):
        evidence=[]
        for source, quote in refs:
            text=sources[source]['text']
            assert quote in text, (q, quote)
            evidence.append({'source':sources[source]['corpus_file'], 'original_path':source,
                             'quote':quote, 'line':text[:text.index(quote)].count('\n')+1})
        cases.append({'question_id':f'{kind}_{sum(c["question_type"]==kind for c in cases)+1:02}',
            'split':'test','question_type':kind,'question':q,'answerable':kind!='unanswerable',
            'reference_answer':answer,'key_facts':[{'fact_id':f'f{i+1}', 'evidence':[e]} for i,e in enumerate(evidence)],
            'evidence':evidence,'allowed_variants':['等义改写可接受；数值、单位、条件和来源关系必须一致'],
            'expected_behavior':behavior,'origin':origin,
            'review':{'author':'AI assistant, evidence-grounded manual construction',
                      'programmatic':'exact quote and source line checked', 'human':'pending'}})
    singles=[
      ('当前中文向量模型的名称是什么？','BAAI/bge-small-zh-v1.5','`BAAI/bge-small-zh-v1.5`，512 维'),
      ('Horus 的图流程用哪个框架编排？','LangGraph','使用 Streamlit 与 LangGraph 编排检索'),
      ('默认负责文档重排序的模型是什么？','BAAI/bge-reranker-base','默认使用 `BAAI/bge-reranker-base`'),
      ('知识库支持哪些输入文件格式？','UTF-8 TXT 和可提取文字的 PDF','支持 UTF-8 TXT 和可提取文字的 PDF'),
      ('用户的纠错文件应放在哪个目录？','data/correction_inbox/','监听 `data/correction_inbox/`'),
      ('默认主生成模型的标识是什么？','deepseek-chat','默认主生成模型是 `deepseek-chat`'),
      ('联网搜索接入哪家服务？','百度千帆','联网：百度千帆搜索'),
      ('构建知识库应该运行什么命令？','python -m src.data_ingestion','python -m src.data_ingestion'),
      ('本机验证使用的 Python 版本是什么？','Python 3.12','本机验证环境为 **Python 3.12**'),
      ('项目采用什么开源许可证？','MIT','[MIT](LICENSE)'),
    ]
    for q,a,e in singles: add('single',q,a,[(R,e)])
    integration=[
      ('新机器要运行本地问答，在启动前需要准备模型和知识库的哪些步骤？','先下载 Embedding 和重排模型，再放入文档并入库。',[(R,'**新机器必须先下载模型**'),(R,'创建 `knowledge_base/`，放入自己的 `.txt` 或 `.pdf` 文件')]),
      ('生成回答和事实核查为什么能使用一致的证据编号？','合并阶段统一编号，生成和核查上下文保持一致。',[(R,'统一证据编号与上下文'),(O,'本地与联网结果的引用编号、显示顺序、核查上下文保持一致。')]),
      ('切换 Embedding 后，知识库和纠偏记忆各需要如何处理？','原文重新入库；纠偏记忆重新向量化并按签名隔离。',[(R,'应选择新索引目录并重新入库'),(O,'纠偏记忆也必须重新向量化')]),
      ('Horus 如何同时利用语义检索和关键词检索，并控制重复证据？','向量和 BM25 召回，稳定 chunk_id 去重并 RRF 融合。',[(R,'向量与 BM25 各召回最多 20 条'),(O,'RRF 每条召回列表内先去重。')]),
      ('如何同时控制外部 LLM 的重试开销和自纠正次数？','SDK 重试关闭，应用控制重试；图默认只重写一次。',[(O,'SDK 内置重试关闭；只由应用配置重试。'),(O,'图统一控制核查循环，默认最多重写一次')]),
      ('本地纯离线评测和线上回答对 API Key 的要求有何区别？','离线检索评测不需要远程 Key；生成至少配 DeepSeek 或 SiliconFlow。',[(R,'TXT/PDF 入库和离线检索评测不调用远程 LLM'),(R,'至少配置 DeepSeek 或 SiliconFlow 中的一路')]),
      ('缓存如何避免把不同对话或旧索引的结果混用？','缓存键含对话偏好索引版本配置，同时设 TTL 容量。',[(O,'缓存键包含对话、偏好、索引版本和配置'),(O,'结果有 TTL 和容量上限')]),
      ('为什么重复入库不会堆积重复文档，写入失败又如何保护旧块？','稳定 ID；成功新增后才删旧块。',[(O,'稳定 ID 使重复入库不增加记录'),(O,'新增向量成功后才删除旧块。')]),
      ('如果原问题没有证据，默认会不会调用 HyDE 或强行生成答案？','默认关闭 HyDE；无证据直接拒答。',[(O,'默认禁用 HyDE 和外部查询改写。'),(O,'无证据时直接拒答')]),
      ('为何仅看关键词命中率和检索耗时不足以判断回答可靠？','关键词命中不能当幻觉率；答案正确与忠实需独立评审。',[(O,'关键词命中率不能充当幻觉率。'),(O,'答案是否正确、是否忠实于证据，需要另行人工标注或独立评审')]),
    ]
    for q,a,e in integration: add('integration',q,a,e)
    constraints=[
      ('默认分块大小和重叠分别是多少，计量单位是什么？','220 / 30 tokenizer tokens',[(R,'默认 220 token、重叠 30 token')]),
      ('混合检索每路候选、重排候选和最终证据上限分别是多少？','20、10、5',[(R,'各召回最多 20 条'),(R,'最多重排 10 条'),(R,'返回最多 5 条有效证据')]),
      ('请求时间预算是多少，是否包含初始化？','60 秒，包含初始化',[(R,'`60` 秒 | 包括初始化的请求时间预算')]),
      ('请求结果缓存的过期时间和容量上限是多少？','300 秒和64项',[(R,'`300` 秒 / `64` | 会话内结果缓存')]),
      ('什么条件下混合模式会转向联网搜索？','无有效本地证据且已配置搜索服务',[(R,'无有效证据且已配置搜索服务时联网')]),
      ('启用 HyDE 后什么时候才补充候选，又按什么问题重排？','原问题无有效证据时；按原问题重排',[(R,'仅在原问题没有有效证据时补充候选，仍按原问题重排')]),
      ('重排器的默认分数门槛是多少，它是不是校准概率？','0.3，不是校准概率',[(R,'`0.3` | sigmoid 重排分数门槛，不是校准概率')]),
      ('纠偏记忆的距离门槛和度量分别是什么？','0.35，平方 L2',[(R,'`0.35` | 纠偏记忆平方 L2 距离上限')]),
      ('默认单次 LLM 输出的 token 上限是多少？','512',[(R,'`512` | 单次 LLM 输出上限')]),
      ('没有 manifest 的旧索引迁移，对模型配置有什么限制？','仅接受原项目 MiniLM 配置',[(O,'旧索引没有 manifest 时，脚本仅接受原项目的 MiniLM 配置。')]),
    ]
    for q,a,e in constraints: add('constraint',q,a,e)
    falsehoods=[
      ('Horus 既然自带 OCR，扫描 PDF 入库会自动识字吧？','错误，需先 OCR，程序不包含 OCR。','当前入库程序不包含 OCR'),
      ('核查输出不是合法 JSON 时也会算核查通过，对吗？','错误，解析失败不算通过。','解析失败不算通过'),
      ('身份类问题也必须加载知识库、检索并核查吗？','错误，身份问题跳过检索核查。','身份类问题直接生成并跳过检索和核查'),
      ('中文向量指令既加在问题上也加在每篇文档上，对吗？','错误，仅查询加指令。','仅查询添加中文检索指令'),
      ('migrate_index.py 可以直接把 MiniLM 向量转换成中文 BGE 吗？','错误，它只能用于相同模型配置的迁移。','不可用它更换 Embedding'),
      ('重排器失效之后就不再做任何相关性筛选了，对吗？','错误，退回词项覆盖过滤。','未启用或加载失败时使用词项覆盖过滤'),
      ('超时能立即杀掉正在运行的本地模型计算吗？','错误，无法强制中断。','无法强制中断已在运行的本地模型计算'),
      ('知识库更新已经是完整的数据库事务发布，对吗？','错误，还不是事务发布。','当前入库不是完整的事务发布流程'),
      ('联网回答的证据来自自动抓取的整篇网页吗？','错误，使用搜索摘要。','搜索结果使用摘要，不包含自动抓取全文'),
      ('这次仓库里的检索评测已经测过真实 LLM 的端到端耗时，对吗？','错误，真实 API 端到端未实测。','真实 LLM/API 的端到端耗时未实测'),
    ]
    for q,a,e in falsehoods: add('false_premise',q,a,[(R,e)],behavior='explicitly_correct_false_premise')
    policies=[
      ('青禾差旅','每日住宿报销上限','600元','800元'),('白鹭请假','年假申请提前时间','3个工作日','5个工作日'),
      ('北辰运维','完整备份间隔','12小时','24小时'),('云帆采购','免审批采购金额上限','2000元','3000元'),
      ('松果客服','首次响应时限','2小时','4小时'),('星桥仓储','库存盘点频次','每周一次','每月一次'),
      ('远山培训','年度必修课时','16小时','24小时'),('清泉设备','借用最长期限','7天','14天'),
      ('海棠合同','档案保存期限','3年','5年'),('银杉访客','来访预约提前时间','1天','2天')]
    for name,field,a,b in policies:
        refs=[]
        for label,value in [('甲',a),('乙',b)]:
            source=f'synthetic/{name}_{label}'
            text=f'【合成评估资料，不是真实业务政策】\n{name}规定（来源{label}）：{field}为{value}。两份规定均未提供生效日期、版本优先级或废止说明。'
            target=source.replace('/','__')+'.txt'
            (corpus/target).write_text(text,encoding='utf-8')
            sources[source]={'corpus_file':target,'text':text,'origin':'synthetic_conflict','sha256':hashlib.sha256(text.encode()).hexdigest()}
            refs.append((source,f'{field}为{value}'))
        add('conflict',f'{name}的{field}究竟是多少？',f'来源甲为{a}，来源乙为{b}；资料冲突且无优先级，无法确定唯一值。',refs,'synthetic_conflict','state_both_values_and_sources_then_uncertainty')
    missing=[('青禾差旅','出差机票舱位标准'),('白鹭请假','产假天数'),('北辰运维','数据库管理员的姓名'),
             ('云帆采购','供应商税号'),('松果客服','客户满意度得分'),('星桥仓储','仓库街道地址'),
             ('远山培训','补考费用'),('清泉设备','逾期罚款金额'),('海棠合同','法务负责人电话'),('银杉访客','停车收费标准')]
    for name,field in missing:
        add('unanswerable',f'{name}的{field}是什么？','现有资料没有该信息，无法回答。',[],'synthetic_missing_attribute','refuse_specific_missing_information_without_guessing')
    assert len(cases)==60
    save('cases.json',cases)
    dev=[]
    for i,(kind,q,answer,text) in enumerate([
      ('single','测试灯塔的颜色是什么？','蓝色','测试灯塔的颜色为蓝色。'),
      ('integration','测试列车从哪里出发、在哪里终到？','甲站出发乙站终到','测试列车从甲站出发。测试列车在乙站终到。'),
      ('constraint','测试图书馆周日几点开放？','上午9点','测试图书馆周日上午9点开放。'),
      ('unanswerable','测试灯塔的建造者是谁？','没有资料','测试灯塔的颜色为蓝色。'),
      ('conflict','测试展厅的票价到底是多少？','两份资料20元和30元，无优先级无法确定','测试展厅资料甲票价20元。测试展厅资料乙票价30元。两者无优先级。'),
      ('false_premise','测试湖泊既然是咸水湖，它的盐度是多少？','前提错，淡水湖','测试湖泊是淡水湖。')]):
        target=f'dev_{i}.txt'; (corpus/target).write_text(text,encoding='utf-8')
        dev.append({'question_id':f'dev_{i+1}', 'split':'dev','question_type':kind,'question':q,'answerable':kind!='unanswerable',
          'reference_answer':answer,'evidence':[] if kind=='unanswerable' else [{'source':target,'quote':text}], 'origin':'synthetic_dev'})
    save('dev_cases.json',dev)
    # Controlled checker challenge: reference labels derive from explicit, single-fact synthetic contexts.
    challenge=[]
    for i,(name,field,a,b) in enumerate(policies):
        ctx=f'{name}规定：{field}为{a}。'
        for label,claim in [('支持',f'{name}的{field}为{a}。'),('矛盾',f'{name}的{field}为{b}。'),('证据不足',f'{name}规定由李明签署。')]:
            challenge.append({'claim_id':f'check_{i+1}_{label}','context':ctx,'claim':claim,'reference_verdict':label,
                'origin':'synthetic_atomic','review':'AI constructed; human pending'})
    save('checker_cases.json',challenge)
    save('sources.json',{k:{x:v for x,v in d.items() if x!='text'} for k,d in sources.items()})
    shutil.copyfile(PROJECT/'config.yaml',ROOT/'config.original.yaml')
    manifest={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(corpus.glob('*.txt'))}
    manifest.update({name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['cases.json','dev_cases.json','checker_cases.json']})
    save('data_manifest.json',manifest)
    print('Created 60 test cases, 6 development cases, 30 checker challenges. Exact source quotes validated; human review pending.')

if __name__=='__main__': main()
