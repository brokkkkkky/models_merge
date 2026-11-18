# Models_Merge

在 Qwen2.5 上做单任务微调，把各任务的权重差（任务向量 τ）合并回一个模型，并用同一套
verbalizer 口径评测合并效果。仓库重点是**融合方法的对照实现**，不是训练框架。

- 底座：`Qwen/Qwen2.5-1.5B`（LoRA 实验）、`Qwen/Qwen2.5-0.5B`（全参实验）
- 任务：GLUE 七件套 `mnli / qnli / rte / qqp / mrpc / sst2 / cola`，外加 `agnews`；
  全部转成"让模型续写一个标签词"的生成式格式
- 融合：Task Arithmetic、DARE + TA、TIES、DARE + TIES、TIES + CMA-ES（逐任务 λ 搜索）

## 目录

```
Models_Merge/
├── common.py                  # 任务定义 / prompt 构造 / 标签词校验 / evaluate()
├── finetune.py                # 单任务微调：LoRA 或 --full_ft
├── eval.py                    # 统一评测：base / base+adapter / 合并后的 checkpoint
├── merge_method/
│   ├── task_arithmetic.py     # θ0 + Σ λi·τi；也提供 AdapterDelta / FullFTDelta
│   ├── dare_ta_merging.py     # DARE + Task Arithmetic
│   ├── ties_merging.py        # TIES-Merging
│   ├── dare_ties_merging.py   # DARE + TIES
│   ├── ties_cma_es.py         # 用 CMA-ES 搜索 TIES 的逐任务 λ
│   └── measure.py             # 融合前的 τ 量级体检
├── requirements.txt
└── record.xlsx                # 手工实验记录（见文末「实验结果」）
```

训练、评测、融合共用 `common.py` 里的模板与 `evaluate()`，避免三处口径漂移。

## 环境

`merge310` 环境（Python 3.10）。Blackwell 显卡（sm_120）必须用 cu128 的 torch：

```powershell
conda activate merge310
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 只有 ties_cma_es.py 需要，requirements.txt 里没有
pip install cma

$env:HF_ENDPOINT = "https://hf-mirror.com"
```

验证 sm_120 可用：

```powershell
python -c "import torch;print(torch.__version__, torch.cuda.get_device_capability())"
# 期望：2.x.x+cu128 (12, 0)
```

代码按 transformers 5.x 写：`dtype=` 取代 `torch_dtype=`，`warmup_ratio` 已移除（自己按 3% 折算成步数）。

## 任务与数据格式

| 任务 | 数据集 | 文本字段 | 标签词（按 label 整数顺序） | 默认 epochs |
| --- | --- | --- | --- | --- |
| `sst2` | `nyu-mll/glue` : sst2 | `sentence` | ` negative` / ` positive` | 1 |
| `agnews` | `fancyzhx/ag_news` | `text` | ` World` / ` Sports` / ` Business` / ` Technology` | 1 |
| `cola` | `nyu-mll/glue` : cola | `sentence` | ` no` / ` yes` | 3 |
| `mnli` | `nyu-mll/glue` : mnli | `premise` + `hypothesis` | ` entail` / ` neutral` / ` contradiction` | 1 |
| `qnli` | `nyu-mll/glue` : qnli | `question` + `sentence` | ` yes` / ` no` | 1 |
| `qqp` | `nyu-mll/glue` : qqp | `question1` + `question2` | ` no` / ` yes` | 1 |
| `rte` | `nyu-mll/glue` : rte | `sentence1` + `sentence2` | ` yes` / ` no` | 10 |
| `mrpc` | `nyu-mll/glue` : mrpc | `sentence1` + `sentence2` | ` no` / ` yes` | 7 |

MNLI 默认用 `validation_matched`，AG News 用 `test`，其余 GLUE 任务用 `validation`。

prompt 长这样（单句 / 双句）：

```
Review: a stirring , funny and finally transporting re-imagining of beauty and the beast
Is the sentiment positive or negative? Answer: positive
```

```
Premise: ...
Hypothesis: ...
What is the relation between the premise and the hypothesis? Answer: entail
```

- 模型只被要求续写一个标签词，评测时取该词的 logit，不做生成。
- 启动时 `check_label_words()` 校验每个标签词恰好 1 个 token，不是就直接报错。
  `entailment`、`paraphrase` 这类会被切碎，所以用小写词干。
- `build_prompt_ids()` 分段 tokenize，超长时只截断正文，前缀和后缀保持完整；默认 `max_len=256`，
  并给答案和 eos 预留 2 个 token。
- 两个踩过的坑写在 `common.py` 注释里：SST-2 必须用完整问句，否则 base 会把 93% 的样本判成同一类
  （0.56 → 0.945）；AG News 第 4 类必须写 `" Technology"` 而不是 `" Tech"`，否则该类几乎不被预测。

## 微调

```powershell
# LoRA（默认模式）
python finetune.py --task mnli
python finetune.py --task rte

# 全参微调：落到独立目录，避免覆盖 LoRA 产物
python finetune.py --task mnli --full_ft --lr 1e-5 --output_dir outputs_ft
```

LoRA 默认超参：`r=16`、`alpha=32`（缩放 2.0）、`dropout=0.05`，作用在 7 类线性层
`q_proj / k_proj / v_proj / o_proj / gate_proj / up_proj / down_proj` 上（1.5B 即 196 个模块）；
`bs=4 × grad_accum=4`、`lr=1e-4`、cosine + 3% warmup、bf16 + gradient checkpointing
（`--no_grad_ckpt` 关掉，快约 25%，多占 1–2 GB 显存）。

epochs 默认值按任务给（见上表），目的是让各任务的训练步数落在同一量级：τ 的范数和步数正相关，
步数差太多会让后续融合隐式加权。

常用参数：`--max_samples`（0 = 全量）、`--data_path` / `--val_data_path`（Hub 拉不动时给本地 parquet）、
`--num_workers`、`--seed`、`--lora_r/--lora_alpha/--lora_dropout`。

产物：

- LoRA → `outputs/<task>/adapter/`（只有 A/B 矩阵，约 40 MB）
- 全参 → `<output_dir>/<task>/model/`（1.5B 约 3 GB，0.5B 约 1 GB）
- 两者都会写 `<output_dir>/<task>/meta.json`（base commit、prompt、超参、验证集分数、库版本）
  和 `eval/<task>/<task>[_ft].json`

## 评测

```powershell
# 1. base 零样本（θ0）
python eval.py --task mnli

# 2. 单任务模型（base + adapter）
python eval.py --task mnli --adapter outputs/mnli/adapter

# 3. 融合后的完整 checkpoint
python eval.py --task mnli --base outputs/merged/ties --tag ties
```

口径全部来自 `common.evaluate()`：

- 只看最后一个 prompt token 之后、各标签词对应位置的 logit，取 argmax；不生成、不比对字符串
- 评测时切到左 padding，才能直接用 `logits[:, -1]`
- 报告 `accuracy`、`mean_nll`、每类准确率、混淆矩阵，以及前 `--show` 条样本的 logits
- `--max_samples` 抽样前先 `shuffle(seed=0)`，避免按标签排序的数据集只取到单一类别
- 结果落盘 `eval/<task>/<tag>.json`；`--tag` 不填时按 adapter 目录名 / base 目录名自动取

## 融合

所有融合脚本共用 `task_arithmetic.py` 里的 `open_source(path)`，按目录里的特征文件自动识别输入：

| 目录里有 | 类型 | τ |
| --- | --- | --- |
| `adapter_model.safetensors` | LoRA adapter | `B @ A * (alpha / r)` |
| `model.safetensors` | 全参 checkpoint | `W_ft - W_base` |

- 分片 checkpoint（`model.safetensors.index.json`）不支持。
- Qwen2.5 的 `tie_word_embeddings=True` 会让 `save_pretrained` 去重 `lm_head.weight`；
  全参路径取交集后自动跳过它（更新 `embed_tokens` 已经等于更新 `lm_head`，不能加两次）。
- 除 `task_arithmetic.py` 需要用 `--full_ft` 显式声明外，其余脚本都自动识别输入类型。
- 融合脚本都把底座以 bf16 加载到 0 号卡（`device_map={"": 0}`），需要 GPU。

| 脚本 | 流程 | 关键超参 |
| --- | --- | --- |
| `task_arithmetic.py` | `θ0 + Σ λi·τi` | `--lams`（默认 1/n） |
| `dare_ta_merging.py` | `θ0 + Σ λi·DARE(τi)` | `--lam` / `--lams`、`--p`、`--seed` |
| `ties_merging.py` | Trim → Elect Sign → Disjoint Merge | `--trim`、`--lam`(1.0) / `--lams` |
| `dare_ties_merging.py` | DARE 之后再做 TIES | `--p`、`--trim`、`--lam` |
| `ties_cma_es.py` | TIES + CMA-ES 搜逐任务 λ | `--pop`、`--gens`、`--sigma`、`--lo`、`--hi` |

先定义两个变量，后面的命令统一复用（逗号分隔，顺序要和 `--lams` / `--tasks` 对齐）：

```powershell
$ADAPTERS = "outputs/mnli/adapter,outputs/qnli/adapter,outputs/rte/adapter,outputs/qqp/adapter,outputs/mrpc/adapter,outputs/sst2/adapter,outputs/cola/adapter"
$FT_ADAPTERS = "outputs_ft/mnli/model,outputs_ft/qnli/model,outputs_ft/rte/model,outputs_ft/qqp/model,outputs_ft/mrpc/model,outputs_ft/sst2/model,outputs_ft/cola/model"
```

### Task Arithmetic

```powershell
# 最简：默认 --adapters = outputs/sst2/adapter,outputs/agnews/adapter，λ 取 1/n
python merge_method/task_arithmetic.py

# 7 个 LoRA adapter 均匀合并 → outputs/merged/avg_7
python merge_method/task_arithmetic.py --adapters $ADAPTERS

# 手写 λ（可为负）→ outputs/merged/lam0.4_...
python merge_method/task_arithmetic.py --adapters $ADAPTERS --lams 0.4,0.4,0.4,0.4,0.4,0.4,0.4

# 0.5B 全参 checkpoint → outputs/merged_ft/lam04
python merge_method/task_arithmetic.py --base Qwen/Qwen2.5-0.5B --full_ft `
  --adapters $FT_ADAPTERS --lams 0.4,0.4,0.4,0.4,0.4,0.4,0.4 --out outputs/merged_ft/lam04
```

输出目录默认 `outputs/merged/avg_N`（均匀，N 是模型数）或
`outputs/merged/lam0.4_0.4_0.4_0.4_0.4_0.4_0.4`（手写 λ，小数截到 3 位），
`--full_ft` 时前缀换成 `outputs/merged_ft`；`--out` 优先。

### DARE + TA

```powershell
python merge_method/dare_ta_merging.py --adapters $ADAPTERS `
  --lam 0.36 --p 0.1 --seed 0 --out outputs/merged/ta_dare_p10_lam036
```

DARE 把保留下来的元素放大 `1/(1-p)`，所以和普通 TA 等幅值的点是：

```
λ_dare = λ_ta * (1 - p)
```

例如 TA 最优 λ=0.4 → p=0.1 用 0.36，p=0.3 用 0.28，p=0.5 用 0.20。
`--lams` 可给逐模型权重（覆盖 `--lam`），不填则默认 1/n。

### TIES

```powershell
python merge_method/ties_merging.py --adapters $ADAPTERS --trim 0.2 --out outputs/merged/ties
```

- `--lam` 默认 **1.0**，不要除以 n：TIES 最后按"同向元素个数"取平均，已经归一化过。
- 需要逐任务加权用 `--lams`，权重作用在 TIES **之前**，会影响符号选举和裁剪——这正是
  `ties_cma_es.py` 要搜的东西。
- `--skip_embed`（跳过 `embed_tokens` / `lm_head`，全参合并省内存）、`--max_modules`（调试）、
  `--dry_run`（只算不存）。

### DARE + TIES

```powershell
python merge_method/dare_ties_merging.py --adapters $ADAPTERS `
  --p 0.1 --trim 0.2 --lam 1.0 --seed 0 --out outputs/merged/dare_ties
```

裁剪按**非零元素**计数（DARE 之后大量元素已经是 0），所以最终保留比例是 `(1-p) × trim`：

| p | trim | 实际保留 |
| --- | --- | --- |
| 0.9 | 0.2 | 2% |
| 0.5 | 0.4 | 20% |

口径与 `record.xlsx` 里的历史实验保持一致，刻意没有改成"占整个张量的比例"。
要和普通 TIES 的 `--trim` 对齐，得把 `trim` 放大 `1/(1-p)`。`--p 0` 等价于退回 TIES。

### TIES + CMA-ES

```powershell
python merge_method/ties_cma_es.py `
  --adapters $ADAPTERS `
  --tasks mnli,qnli,rte,qqp,mrpc,sst2,cola --trim 0.2 `
  --pop 5 --gens 6 --sigma 0.3 --lo 0.3 --hi 2.0 `
  --eval_max_samples 300 --holdout cola --topk 3 `
  --out outputs/merged/ties_cma --log eval/ties_cma.json
```

- 模型只加载一次，`apply()` 在显存里改权重，`restore()` 用 CPU 上的 baseline 快照精确回滚
  （用 `w - τ` 回滚会残留 bf16 舍入误差，几十次后累积）。
- 目标函数 = 除 `--holdout` 外各任务准确率的平均；`--eval_max_samples 300` 是代理评测，
  搜完用全量复核 top-k 再选最优。
- λ 顺序与 `--adapters` / `--tasks` 一致；只有给了 `--out` / `--log` 才落盘。

### 融合前体检

```powershell
python merge_method/measure.py                                    # LoRA（outputs/）
python merge_method/measure.py --full_ft --base Qwen/Qwen2.5-0.5B  # 全参（outputs_ft/）
```

按任务打印 `|τ|` 的最大值、p99.9、p99、均值和非零占比。DARE 的前提是 τ 足够稠密且大量元素
接近 0；LoRA 的低秩 τ 往往不满足这个前提，先看这张表再决定要不要开 DARE。

## 产物

```
outputs/<task>/adapter/        # LoRA A/B 矩阵，约 40 MB
outputs/<task>/meta.json       # base commit、prompt、超参、验证集分数、库版本
outputs/<task>/run/            # Trainer 临时目录（save_strategy=no，不留 checkpoint）
outputs_ft/<task>/model/       # 全参 checkpoint，1.5B ≈ 3 GB / 0.5B ≈ 1 GB
outputs/merged/<tag>/          # 合并后的完整模型，每个 3 GB+
outputs/merged_ft/<tag>/
eval/<task>/<tag>.json         # 每次评测的完整结果
```

`.gitignore` 已排除 `outputs/`、`outputs_ft/`、`eval/` 以及 `*.safetensors` / `*.bin` / `*.pt`。

## 实验结果

原始记录在 `record.xlsx`。**0.5B 全参 + GLUE 七任务**（验证集全量）：
