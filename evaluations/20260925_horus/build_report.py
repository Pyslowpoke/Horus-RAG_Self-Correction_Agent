"""Build human-readable report from this run's immutable measurements and AI review."""
from pathlib import Path
import json,csv,collections
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(str(x) for x in r)+' |' for r in rows])
def pct(x):return 'NA' if x is None else f'{100*x:.1f}%'
def main():
    paid=ROOT/'runs/paid_test_01';local=ROOT/'runs/local_01'
    ss=read(paid/'answer_summary.json');summary={r['arm']:r for r in ss if r['group']=='overall'}
    ls={r['arm']:r for r in read(local/'summary.json') if r['group']=='overall'}
    usage=read(ROOT/'api_usage_final.json');matrix=read(paid/'checker_confusion.json')
    retrieval=table(['指标（50有答案/10无答案）','A 纯向量','B 混合+重排过滤','V 向量+同重排过滤'],[
      ['标注原文任一证据命中',*[f"{round(ls[k]['any_hit']*50)}/50" for k in 'ABV']],
      ['标注原文完整证据命中',*[f"{round(ls[k]['all_hit']*50)}/50" for k in 'ABV']],
      ['证据单元召回（宏平均）',*[pct(ls[k]['evidence_recall']) for k in 'ABV']],
      ['MRR',*[f"{ls[k]['mrr']:.3f}" for k in 'ABV']],
      ['标注块Precision@5',*[pct(ls[k]['precision_at_5']) for k in 'ABV']],
      ['返回块标注精度',*[pct(ls[k]['returned_precision']) for k in 'ABV']],
      ['无答案题返回空证据',*[f"{round(ls[k]['negative_empty_rate']*10)}/10" for k in 'ABV']],
      ['有答案题返回空证据',*[f"{round(ls[k]['positive_empty_rate']*50)}/50" for k in 'ABV']],
      ['检索P50 / P95（秒）',*[f"{ls[k]['p50_seconds']:.4f} / {ls[k]['p95_seconds']:.4f}" for k in 'ABV']]])
    answers=table(['指标','A','B','C（B+核查纠正）'],[
      ['完全正确（AI评审）',*[f"{summary[k]['fully_correct']}/60 ({pct(summary[k]['fully_correct_rate'])})" for k in 'ABC']],
      ['正确性均分（正确1/部分0.5/错误0）',*[f"{summary[k]['correctness_score']:.3f}" for k in 'ABC']],
      ['关键信息完整性（50题宏平均）',*[pct(summary[k]['completeness_macro']) for k in 'ABC']],
      ['有答案题整体误拒答',*[f"{round(summary[k]['false_refusal_rate']*50)}/50" for k in 'ABC']],
      ['无答案题正确拒答',*[f"{round(summary[k]['correct_refusal_rate']*10)}/10" for k in 'ABC']],
      ['冲突题正确处理',*[f"{round(summary[k]['conflict_correct_rate']*10)}/10" for k in 'ABC']],
      ['答案交付失败',*[f"{summary[k]['answer_failures']}/60" for k in 'ABC']],
      ['预期工作流失败（含核查错误）',*[f"{summary[k]['workflow_failures']}/60" for k in 'ABC']],
      ['工作流失败按0的均分',*[f"{summary[k]['availability_adjusted_score']:.3f}" for k in 'ABC']],
      ['仅工作流成功样本均分',*[f"{summary[k]['success_only_score']:.3f} (n={summary[k]['successful_workflows']})" for k in 'ABC']]])
    bytype=table(['题型/层（每方案样本数）','A完全正确','B完全正确','C完全正确'],[
       [name+f' (n={n})',*[f"{next(r['fully_correct'] for r in ss if r['group']==group and r['arm']==k)}/{n}" for k in 'ABC']]
       for group,name,n in [('single','单一事实',10),('integration','整合',10),('constraint','条件/数字',10),('unanswerable','无答案',10),('conflict','冲突',10),('false_premise','错误前提',10),('repository_natural','项目自然资料',40),('synthetic','合成资料',20)]])
    fidelity=table(['独立AI审阅的陈述/引用指标','A','B','C'],[
      ['事实陈述数',*[summary[k]['factual_claims'] for k in 'ABC']],
      ['支持 / 矛盾 / 证据不足（条）',*[f"{summary[k]['supported']} / {summary[k]['contradicted']} / {summary[k]['insufficient']}" for k in 'ABC']],
      ['支持率',*[pct(summary[k]['supported_rate']) for k in 'ABC']],
      ['矛盾率',*[pct(summary[k]['contradicted_rate']) for k in 'ABC']],
      ['证据不足率',*[pct(summary[k]['insufficient_rate']) for k in 'ABC']],
      ['事实陈述有数字引用',*[pct(summary[k]['citation_presence_rate']) for k in 'ABC']],
      ['出现的数字引用指向存在编号',*[pct(summary[k]['citation_existence_rate']) for k in 'ABC']],
      ['带引用事实获得所引资料支持',*[pct(summary[k]['citation_support_rate']) for k in 'ABC']]])
    costs=table(['每方案60题','A','B','C（包含B首次生成）'],[
      ['实际API调用次数',*[summary[k]['api_calls_including_B_for_C'] for k in 'ABC']],
      ['已知输入Token',*[summary[k]['input_tokens_known'] for k in 'ABC']],
      ['已知输出Token',*[summary[k]['output_tokens_known'] for k in 'ABC']],
      ['usage缺失的调用数',*[summary[k]['calls_missing_usage'] for k in 'ABC']],
      ['固定证据图P50 / P95（秒）',*[f"{summary[k]['fixed_evidence_p50_seconds']:.2f} / {summary[k]['fixed_evidence_p95_seconds']:.2f}" for k in 'ABC']],
      ['检索+图组件合成P50 / P95（秒）',*[f"{summary[k]['composed_p50_seconds']:.2f} / {summary[k]['composed_p95_seconds']:.2f}" for k in 'ABC']]])
    confusion=table(['独立参考\\产品预测','支持','矛盾','证据不足','错误/缺失'],[
        [label,*[matrix['matrix'][label][c] for c in ['支持','矛盾','证据不足','missing_or_error']]] for label in ['支持','矛盾','证据不足']])
    classtable=table(['类别','Precision','Recall','F1'],[[r['label'],pct(r['precision']),pct(r['recall']),f"{r['f1']:.3f}"] for r in matrix['per_class']])
    text=f'''# Horus 可复现效果评估报告

日期：2026-09-25。主代码：`5616fc9afe55b38e135365c8448e373d7900e832`。报告仅使用本目录本次运行的数据。

## 结论

**可以作为AI产品应届生的工程原型与评测项目写入简历；当前不支持“事实核查显著提高正确率/降低幻觉”的效果宣传，也没有生产可用性证据。** 混合检索在这套小题集上有样本内收益和明显CPU开销，统计区间仍容许没有提升；核查纠正没有新增正确答案，且引入格式错误、超时和大量额外调用。

建议保留核查的可观测接口，**当前自动重写默认启用不划算**。先修复核查可靠性和失败回退，再用新的独立问题集验证按条件启用。这个建议是基于本次样本的工程判断，不是证明所有事实核查都无效。

## 1. 已完成与未完成

- 实际运行36项产品回归，全部通过（模型/网络为替身）；另6项评估指标测试通过。
- 固定公开项目资料和合成资料，建立100个分块的独立Chroma索引；60题×A/B/V×3轮，共540次正式真实本地检索，0失败、0降级，所有180个“题×方案”在3轮返回ID及分数指标一致。
- 6题开发试跑真实20次API调用；正式60题A/B/C产生180个方案记录，加30条核查挑战，实际223次API调用。未因失败补跑或替换结果。
- 137份唯一匿名“问题+答案+上下文”经本次编程助手AI逐条审阅，映射回180个回答，保留原始判断和理由；**没有人工评审**。180条人工复核表及22条分层样本已准备，全部待人工填写。
- 完成6个程序契约探针；12个检索失败案例的候选重排事后复算与原选择完全一致，仅作定位，不替换正式结果。
- 未完成：真实联网探索、私人知识库评估、D无资料生成、真实UI单次端到端计时、LLM多轮随机重复、人工独立复核、自然回答所有核查陈述的一一对齐混淆矩阵。前两项避免动态资料与私人数据外传；D非核心；其余受本次范围/预算和无人工复核条件限制。没有声称完成这些项。

## 2. 公平性与可复现边界

正式题六类各10；其中40题来自项目公开说明/代码，10题合成双来源冲突，10题合成缺失属性。开发6题单独记录，其文档也作为固定语料中的干扰项。没有把开发题算入正式结果。合成政策虽覆盖不同字段，但冲突/无答案结构同质，难度偏低；项目文档题也不能代表开放研究任务。

来源、原文、行号、参考答案、允许变体和预期处理在 `cases.json`。AI构建题目并核对原文，程序验证引用及分块覆盖；这不是人类金标准。源码和资料哈希已保存，原私人知识库和历史记忆均未读取入实验或上传。

A是纯向量Top-5；B是现有向量+BM25/RRF+重排/过滤；V是向量Top-10加相同重排/过滤。B−A比较整套检索配置，不把差异只归因于BM25；B−V才更接近候选融合差异。A没有拒答阈值，因此“空证据率”有结构性差异，不能与回答拒答率混用。

A/B使用相同生成模型配置、Prompt、temperature=0.1、max_tokens=512。C逐字复用B初始答案及上下文，只增加核查/最多一次重写/再核查，因此C−B没有再次随机生成的混淆。核查与重写使用另一轻量模型，是当前组合机制贡献，不是同模型纯算法消融。

请求生成模型为 `deepseek-chat`，本次成功响应均标识 `deepseek-flash`，fingerprint一致；核查模型为 `Qwen/Qwen2.5-7B-Instruct`，无可用精确快照fingerprint。供应商别名不等于永久版本锁定。无备用切换、无SDK/应用额外重试、并发1。只运行一轮真实API，网络失败不能被解释为算法能力差异。

结果缓存关闭，模型与索引热实例复用，问题和检索方案顺序固定随机种子。CPU线程4。冷初始化（含入库）72.06秒，18次开发检索预热合计41.21秒，均不算正式检索延迟。桌面机器未做硬件隔离，后台进程/温度等可能影响时间；三轮B的P95分别约5.01、2.32、1.76秒，时延波动需要保留。

## 3. 检索结果

{retrieval}

完整证据命中B−A为+12个百分点，按50题配对bootstrap的95%区间为[0,+26]个百分点；B−V为+4个百分点，区间[0,+10]。不能声称已证明稳定泛化收益。三轮只用于检查稳定性，独立样本仍是60题，不是180题。

这些是**指定来源的标注原文命中**。同一事实在README、代码、配置中重复，未穷尽全部等价证据会低估召回；例如A可从配置答对模型名，却未命中README标注片段。不要把这张表直接称为语义相关度真值。引用与回答另行独立审阅。

B对5/10无答案题返回空证据，但A/B/C最终均对10/10无答案题正确拒答：这明确说明“返回空文档”和“正确拒答”并非同一个指标。B还对5/50有答案题过滤为空，另外2题虽有文档仍拒答。

## 4. 回答质量与题型

**下列语义分数是AI评审，尚未人工复核。** 同一个编程助手参与数据构建和评分，存在作者偏差；匿名输入隐藏方案名，但不能保证完全不可推断。计分规则见 `SCORING_RUBRIC.md`，137条原始判断见 `record_ai_reviews.py` 及 `ai_judgments.json`。

{answers}

{bytype}

A有4题部分正确，其余错误主要是资料缺失后的拒答。B比A完全正确多5题，但其中1题是A生成超时、B成功，不能归因于检索改进。B−A完全正确率差+8.33个百分点，95%配对区间[-3.33,+20.00]；正确性均分差+0.050，区间[-0.050,+0.158]。C−B完全正确率差−1.67个百分点，区间[-5.00,0]，该下降来自一次重写超时。

成功样本均分的分母不同，不能直接把条件成功均分用作公平方案排序。共同交付答案的59题中，C与B的正确性评分全部相同。自然资料40题中B/C都是33题完全正确，不能用20题模板式合成结果掩盖自然资料限制。

## 5. 忠实度、引用与完整性

{fidelity}

纯拒答没有事实陈述时，单题忠实度为NA，不计成100%。表中支持率分母是实际审阅的事实陈述总数；不是所有问题数，也不是“幻觉率=1−正确率”。B/C未发现矛盾或不足陈述，**不代表系统零幻觉**：题集简单、生成很保守、拒答多，而且AI评审没有人类校验。A的3条不足主要是无证据的因果解释和漏掉HyDE启用前提，不能直接称作现实事实错误。

引用存在率用正则与编号表检查；引用语义支持由AI核对。部分答案只是重复粘贴资料，仍可能有很高引用/忠实度，但不增加回答价值。B/C答案完整性仅86%/84%，这比孤立展示100%支持率更能反映实际体验。对于“开头说无法回答、后文实际答出”的回答，按完整内容判断，不用关键词机械判拒答。

## 6. 核查准确性与纠正收益

### 独立原子核查挑战

{confusion}

{classtable}

总计28/30正确（93.3%），其中一条矛盾题超时，一条“资料没给签署人”的陈述被误标为矛盾。后者说明模块可能混淆“证据不足”和“与资料冲突”。这些是先固定参考标签的简单合成原子陈述，不能代表自然回答长文本核查精度或陈述覆盖率。

### 主链路表现

C最终状态：passed 13、failed 13、error 23、skipped 10、整图异常1。需要核查的50题中24题流程出错（48%）；占全部60题40%。主链路66次核查调用中，43次产生可解析条目、16次非法JSON、7次API超时；另有1次重写超时。16次非法JSON中10次达到512输出上限，其余也有说明文字或不合法格式。不能认为仅增大输出上限就能解决全部问题。

在43次可解析的主链路核查中，共48条原标签因“evidence不是上下文中的连续原文”被程序降为证据不足，分布于25次调用。这个计数不等于48次已证明误判，但案例显示复制答案作为证据、跨来源拼接、加入引用编号会把本来支持的事实误判为不足。此外，核查器有时核查了资料里的句子而非原回答的句子，触发无必要重写。

纠正前后：错→对0、对→错1（调用失败无答案）、其余59题分数不变。17题尝试重写，初始B答案的AI评分全部为正确；16题最终仍正确，1题重写超时无最终答案。**没有观察到成功重写带来的语义改好，也没有观察到成功返回文本的语义改坏。** 不能用这一次超时宣称重写模型会把事实改错；可以确认当前异常传播损害了可用性。

## 7. 耗时与成本

{costs}

C新增83次调用、已知输入33,741 Token、输出13,812 Token。上表C已含B的50次首次生成，因此不能再把A+B+C相加当本次账单。正式核查挑战另30次；实际账单口径为正式223次+开发20次=243次，已知输入100,131、输出21,061 Token。10次超时没有usage，不能当作零计费；没有确认的单价，不给货币成本。重试0，模型备用切换0。

**时延不是完整UI端到端。** 固定证据图实测生成/核查/修正；组件合成值再加事先测得的检索时间。C的图时间包含B生成和真实核查增量。其P50从B的1.13秒增至9.79秒，组件合成P50从2.08秒增至10.85秒；没有相应的回答正确性新增收益。服务端自动提示缓存未被控制，usage只记录标准输入/输出，不能拆分缓存计价。

## 8. 代表案例（观察事实与推测分开）

| 案例 | 观察到的结果 | 定位及边界 |
|---|---|---|
| integration_07 缓存隔离 | A只答TTL/容量（0.5）；B答缓存键与限制（1）；C仍为1但附加无关说明 | 混合召回带来完整证据；核查又把上下文里不在B回答中的线程池等句子列为待核查对象。对象漂移是可见事实，Prompt分隔不足是原因推测 |
| integration_08 幂等入库 | A把token分块当作不重复的原因（0.5）；B答稳定ID与先增后删（1）；C保留B但核查超时 | B检索改善答案；核查未提供增益。因果解释不能仅因同段共现就认定有依据 |
| single_03 重排模型名 | A从配置答对；B/C空证据拒答 | 正确README段已进入B重排候选，事后复算分数0.0203低于0.3；这是过滤阶段损失，不能简单归咎Embedding |
| constraint_04 缓存数值 | A答300秒/64条；B/C仅有“存在TTL容量”的片段，拒答具体数字 | 带具体数字的候选重排分数0.1768被过滤；对数字题需校准覆盖与拒答门槛。复算12题结果均与原选择一致 |
| unanswerable_01 机票舱位 | 三方案均正确拒答；C重写后增加住宿政策整段复述 | 资料只有住宿，没有机票舱位；忠实拒答正确，但额外核查未改善答案 |
| conflict_01 住宿上限冲突 | B/C均保留600元与800元且不选唯一值；C仍显示failed | 核查证据字符串混入引用编号或跨来源组合，非原文连续片段；这是引用匹配/核查误报警，不能把“不足”当业务事实错误 |
| conflict_06 盘点频次 | B正确列每周/每月；C重写请求超时，整图无最终答案 | 这是实际可用性损害，不是成功生成的文本语义改坏。应保留已存在的B答案并显示未完成核查 |
| integration_04 混合检索原理 | A/B/C均未答出；B只留下项目简介 | 含向量/BM25说明的候选复算分数0.0370被过滤，另一个RRF证据未进入重排候选。只重写现有上下文无法补齐漏检证据 |
| single_04 文件格式 | B简洁正确；C首句只写TXT，后文又确认TXT和PDF并粘贴资料，整体仍给1 | 是冗余和表达含混，按全文并未丢失PDF事实；不把它人为算成纠正损害。列入人工复核重点 |

原文、候选、所有核查轮次和修改前后答案都可按question_id在JSONL中定位。不存在“真实纠正错变对”的代表案例，故不编造。缺少完整自然核查混淆矩阵也意味着上述错误类型不能推算全部自然陈述误判率。

## 9. 是否值得保留与修复优先级

当前没有一个题型已证明开启核查会提高正确率。建议保留模块供旁路诊断与后续实验，默认避免无条件自动重写。简单单事实查找和已经合理拒答的题，本次看不到支付额外开销的理由。跨来源数字/冲突是可考虑的未来验证场景，但本次B已能正确处理多数，不能声称这些场景已经证明核查有效。

最值得修复的三个问题：

1. **核查输出与对象不可靠**：先固定待核查的回答陈述及ID，再要求结构化schema；不要让模型把参考资料的其他事实加入核查列表。为核查单独设置并验证输出预算，保存解析失败原文；用新开发集验证格式及覆盖率。
2. **证据匹配错误触发重写，失败又丢原答案**：用可追踪的证据片段ID/跨度和多证据支持，区分抽取失败与事实矛盾；不要直接放宽校验让伪造引文通过。重写失败保留B答案与“未完成核查”状态，只有可验证改进才替换。证据缺失应补检索，不靠反复改写补事实。
3. **重排门槛未按题型校准、证据覆盖不足**：在新开发集检查数字、否定前提、技术标识及跨段落覆盖；分析标题/项目实体信息在分块时丢失的影响。当前0.3不是校准概率；不要在这60道正式题上反复调参后再报告“提升”。

本次仅评估并新增独立工具，没有修改上述产品行为。

## 10. 简历结论与建议表述

**可以写，但亮点应是需求拆解、RAG工作流实现、可复现实验、质量/延迟/成本取舍与问题定位。** 当前证据不足以写“消除幻觉”“事实核查提高准确率”“生产级AI研究助手”。

只描述已完成工作：

> 设计并实现 Horus 事实核查型 RAG 原型，串联中文混合检索、证据引用、核查与有限纠正；建立可追溯评估集和对照实验，分析质量、时延与模型调用开销，定位检索过滤及核查可靠性问题。

可用的量化表述仅限本次样本内观测（不要改写为生产提升承诺）：

> 构建60题离线评测与30条核查挑战，完成纯向量、混合检索及核查纠正对照；在50道有答案题上观察到标注原文完整命中32/50与38/50，并识别核查格式错误和超时开销。结果限项目文档及合成场景，未验证生产效果。

由于统计区间宽、AI评审待人工校验，暂不建议把A/B回答正确率差写成确定的优化成果；尤其不能把C的模块接入写成已验证的准确率提升。README修订建议见 `README_claims.md`。

## 11. 局限与审计入口

自然资料是项目自述/源码，40题中多题共享片段；合成20题较模板化；不是随机抽取的业务样本。bootstrap按题重采样，未校正同来源相关性，置信区间只作描述。语义评分由数据作者AI完成，精确评分模型快照不可取得，虽然匿名但非严格外部盲审。真实API只跑一次，服务别名、网络与服务器缓存不可完全固定。没有独立人工复核、公开互联网效果或完整UI端到端数据。

主要文件：`PROTOCOL.md`（口径），`cases.json`（标签），`runs/local_01/retrieval.jsonl`（检索），`runs/paid_test_01/answers.jsonl`（原始回答/核查），`calls.jsonl`（请求响应usage），`ai_judgments.json`（AI评分），`answer_summary.csv`（分层汇总），`answer_paired.csv`（配对区间），`checker_confusion.json`（独立挑战），`human_review_sample.csv`（待人工复核）。旧的占位状态文件不是最终状态，见 `AMENDMENTS.md` 与 `FINAL_STATUS.json`。
'''
    with (ROOT/'REPORT.md').open('x',encoding='utf-8') as f:f.write(text)
    final={'complete':True,'product_regressions_passed':36,'evaluation_metric_tests_passed':6,
      'formal_questions':60,'dev_questions':6,'retrieval_requests':540,'answer_records':180,'checker_challenges':30,
      'unique_blind_AI_reviews':137,'human_reviews_completed':0,'human_review_sample_rows':22,
      'usage':usage['total'],'paid_pilot_cap':26,'paid_formal_cap':290,
      'not_measured':['live_web','private_corpus','D_no_retrieval','full_UI_E2E','LLM_repeats','human_review','full_natural_claim_alignment'],
      'product_code_changed':False,'verdict':'Resume-safe as engineering and evaluation prototype; no demonstrated fact-checking accuracy gain.'}
    with (ROOT/'FINAL_STATUS.json').open('x',encoding='utf-8') as f:json.dump(final,f,ensure_ascii=False,indent=2)
    print('Report and final status written')
if __name__=='__main__':main()
