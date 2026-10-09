# 真实用户试用 / User trial

先招募 5 位确实每周处理此类任务的人，每人带一个可以脱敏的任务。招募消息不要求加星。

Horus：用两份制度/产品说明查一个明确事实，再问资料冲突。IntentLens：用一张脱敏销售表比较指标，再修改口径并下载图表。先看对方独立操作，卡住再协助；不要代替用户完成任务。

记录：渠道与日期、用户角色、任务、原工具、是否开始/完成、等待与卡点、修改次数、是否愿意下周再次用及原因。表格使用匿名编号，不记录真实姓名或业务明细。7 天后问是否实际复用；意愿不是留存。小样本报人数，不能包装百分比。

产品反馈按钮打开 GitHub Issue 表单，需要 GitHub 账号且内容公开。非 GitHub 用户使用访谈记录，由维护者脱敏整理；未经同意不公开。不要要求上传客户资料、密钥、完整日志。

推广分两轮：先定向邀请同学/同事中真实目标用户；核心任务稳定后再发任务视频与复现实例。中文材料讲一次真实卡点、口径和结果，英文材料提供可运行输入、明确限制和 failure case。可面向技术社区/数据分析社区发帖，遵守各社区推广规则，不批量私信，不制造虚假使用证言。

公开部署真实模型前，需要请求限额、并发隔离、费用上限、密钥管理和数据删除能力。当前本地单用户版本不能直接暴露到公网。Horus 可评估 Docker Space，IntentLens 需同时托管 API、worker 与持久存储；静态 Pages 只能演示浏览器逻辑。没有预算和托管账号时先开展陪同试用。

## 招募文案草稿

中文：我在测试一个帮助【查证资料/分析表格】的工具，想找 5 位有实际任务的人试一次。约 15 分钟，使用公开或脱敏数据即可。最想知道你在哪一步卡住、结果是否能用；不需要加星。

English: Looking for five people who regularly work with documents or business spreadsheets to try one concrete task. Bring public or sanitized data; a 15-minute session is enough. I want to learn where the workflow fails and whether the result is usable. No star request.

GitHub feedback forms: https://docs.github.com/en/issues/tracking-your-work-with-issues/learning-about-issues/about-issues
Hosting reference: https://huggingface.co/docs/hub/en/spaces-overview
