# Horus 可复现效果评估 — 2026-09-25

先阅读 [最终报告](REPORT.md) 与 [评估方案](PROTOCOL.md)。本目录保留本次真实运行数据；没有覆盖 `.evaluation/` 中的历史记录。源码审计见 [AUDIT.md](AUDIT.md)，预算/方法补充见 [AMENDMENTS.md](AMENDMENTS.md)。

## 内容

- `cases.json`：60道正式题，参考答案、类型、原文出处、变体和预期行为。
- `dev_cases.json`：6道独立开发题；`checker_cases.json`：30条原子核查挑战。
- `corpus/`、`sources.json`、`data_manifest.json`：公开项目语料及合成资料、来源、哈希。
- `required_facts.json`、`SCORING_RUBRIC.md`：预先参考答案的原子事实及固定评分规则。
- `config.original.yaml`、`environment.json`、`requirements.freeze.txt`、`product_code_manifest.json`：配置、环境与代码指纹。
- `runs/local_01/`：真实检索原始JSONL、全部分块、候选与分数、三轮结果、汇总及配对区间。
- `runs/paid_dev_01/`：开发题原始回答、API提示/响应/usage及错误日志。
- `runs/paid_test_01/`：正式回答、核查/修正、API日志、匿名评分输入与判断、汇总CSV和核查混淆矩阵。
- `human_review.csv` 与正式人工复核表：状态均为待人工复核；AI评分不能代替人工签字。
- `checker_contract_probes.json`：预设LLM输出的程序契约探针，不是模型准确率。
- `README_claims.md`：README与简历表述建议。

只记录请求内容（公开/合成资料）、生成输出、模型名、耗时与usage；不包含认证信息。向量索引为可重建运行产物，已在本目录 `.gitignore` 排除。

## 重运行命令

在项目根目录执行；Windows示例使用已验证的项目虚拟环境。先确认源码仍为报告中的版本，并安装 `requirements.freeze.txt` 对应依赖、本地缓存两个BGE模型。完整锁定文件记录本机环境，其他操作系统的二进制包安装可能需要调整。

```powershell
# 产品回归（使用替身网络/模型，不消耗API）
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
# 评估计算逻辑测试
.\.venv\Scripts\python.exe -m unittest discover -s evaluations/20260925_horus -p test_evaluation.py -v
# 在新目录构建独立索引，先开发题预热，再正式三轮
.\.venv\Scripts\python.exe evaluations/20260925_horus/run_local.py --run-id local_reproduction_01 --repeats 3
.\.venv\Scripts\python.exe evaluations/20260925_horus/analyze_local.py --run-id local_reproduction_01
```

所有原始运行目录都拒绝覆盖。`prepare.py` 用来从源代码版本首次建立数据，现有数据已冻结，无需再次执行。将数据复制到新目录再重建或修改时，必须使用新版本标识，不能覆写正式题。

## 真实API阶段（必须有明确预算授权）

凭证在本机安全环境变量中配置；如已获得读取本机认证配置供客户端使用的授权，可显式加 `--load-project-env`。不要把密钥写进命令行、版本控制或聊天。

```powershell
# 先少量开发题，已有授权上限26次
.\.venv\Scripts\python.exe evaluations/20260925_horus/run_paid.py --split dev --run-id paid_dev_reproduction_01 --retrieval-run local_reproduction_01 --max-calls 26 --budget-authorized --load-project-env
# 看实际调用日志，重新估算并获得正式预算；PAID_PILOT_REVIEWED.json记录本次已获批准的运行
.\.venv\Scripts\python.exe evaluations/20260925_horus/run_paid.py --split test --run-id paid_test_reproduction_01 --retrieval-run local_reproduction_01 --max-calls 290 --budget-authorized --load-project-env
```

上面的开关不会替使用者自动获得预算；本对话的授权仅覆盖本次已执行批次。脚本硬限制请求数和每次输出512 token，输入token和货币费用没有绝对上限；正式新批次仍应重新确认预算。只用固定两个模型，禁用动态备用模型和重试。出现连续3次API失败停止，并保留未执行记录。

## 评分与汇总

```powershell
.\.venv\Scripts\python.exe evaluations/20260925_horus/blind_export.py --run-id paid_test_01
# 由真实评分者按SCORING_RUBRIC逐条评分，保存ai_judgments.json；不能由产品核查器自证
.\.venv\Scripts\python.exe evaluations/20260925_horus/analyze_paid.py --run-id paid_test_01
```

本次上述派生文件已经存在，重复分析也拒绝覆盖。匿名阅读工具 `view_blind.py --start 0 --count 15` 供审计本次输入；映射表保存可追溯性，评分时不应查看。复现AI语义判断不保证逐字一致，因此保留原始判断及理由，另留人工复核。

核查挑战混淆矩阵可直接对预先标签复算；主实验答案、忠实度和引用语义需独立审阅，不能仅靠字符串程序自动给出。汇总仅适用于本次题集和单次API运行，不能代表生产研究助手效果。
