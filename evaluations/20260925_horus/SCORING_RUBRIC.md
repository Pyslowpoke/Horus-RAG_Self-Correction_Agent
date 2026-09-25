# 固定 AI 评分指令与记录格式

评分者：本次对话中的 Codex 编程助手（AI；精确服务模型快照不可读取），不同于项目生成端 deepseek-flash 和核查端 Qwen2.5-7B。不是人工评审。评分者也是数据构建者，存在作者偏差；同一对话已见程序结构，匿名输入只能隐藏方案标签，无法保证完全盲法。不得声称独立人类标注或客观真值。

## 评分 Prompt

你将收到匿名 answer_id、问题、预先固定的参考答案/原文及这次生成实际收到的编号资料。看不到 A/B/C 标签和核查器判定。依据参考答案判正确性，依据实际编号资料判忠实度，两者不要混淆。完全相同问题、答案和上下文只评分一次，评分映射回所有相同记录。

1. 按 PROTOCOL 的 1/0.5/0 规则评答案，关键事实、条件、数字、否定与冲突保留均需核对。实际资料缺失导致合理拒答，可以高忠实但答案正确性为0（问题在完整库可回答）。
2. 将回答拆为最小可核验事实，逐条列原文或忠实压缩，分类 supported / contradicted / insufficient。一般礼貌语和纯粹“无法回答”不算事实；“资料没有提到某属性”为资料范围陈述，可核验，不能一概算现实世界事实错误。复合事实需拆分，重复陈述只计一次。
3. 每条标出引用数字；有效数字由程序核验。有引用只说明标注存在，必须按对应段落检查支持。存在但不蕴含结论的引用支持=false；引用范围要按语句实际指向判断。无引用写[]，不自动扣答案正确分。
4. 完整性针对预先参考答案实际被问及的关键事实。key_facts中的证据可能带额外信息，未问及的附带维度/版本不要求回答。使用 required_facts.json 的原子事实清单记录 covered_fact_indices（从1起）。无答案题无完整性分母。
5. refused 表示对本题要求的关键答案整体拒绝给出，部分回答但有核心事实时不算整体拒答。conflict_handled 只对冲突题填写，需同时给两方值并保留不确定性。
6. 不知道或证据关系不明确时在 notes 标争议，不为了得到整齐指标硬判成功。所有评分留待人工复核。

## 每条原始 AI 判断 JSON

```json
{
  "answer_id": "anon_001",
  "correctness": 1,
  "covered_fact_indices": [1],
  "refused": false,
  "conflict_handled": null,
  "claims": [
    {"text": "回答中的原子事实", "verdict": "supported", "citations": [1], "citation_support": true}
  ],
  "notes": "依据与争议；不是人工判断",
  "reviewer": "Codex conversation AI; exact snapshot unavailable",
  "human_review": "pending"
}
```

程序校验：answer_id全覆盖无重复；correctness取值合法；引用编号存在率直接从回答正则提取；陈述分类合法；covered_fact_indices不越界。统计脚本不会从核查器 verdict 推导评分。核查挑战参考答案在运行前固定，三类混淆矩阵可直接程序计算；如果一个原子陈述被拆成多个冲突标签或返回空列表，记 missing，不取多数标签掩盖歧义。
