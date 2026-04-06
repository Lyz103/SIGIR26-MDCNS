# MDCNS：面向序列推荐的多源负采样框架

> SIGIR 2026 论文代码：**Divergence Meets Consensus: A Multi-Source Negative Sampling Framework for Sequential Recommendation**

## ✨ 项目简介

本仓库提供 MDCNS 的实验代码，核心思想是将多个推荐骨干模型的**差异性**与**共识性**结合起来，用于更有效的负采样与训练优化。

仓库主要包含两部分：

- `MDCNS_Code/`：MDCNS 主方法实现与实验脚本
- `Baselines/`：若干负采样 baseline 的独立实现

## 📁 仓库结构

```text
SIGIR26-MDCNS/
├── MDCNS_Code/          # 主方法代码、数据、日志与输出
├── Baselines/           # Baseline 方法实现
├── SR_CR.pdf            # 论文/材料文件
├── README.md            # 导航页
├── README.zh.md         # 中文说明
└── README.en.md         # English README
```

`MDCNS_Code/` 下的关键内容：

- `main.py`：主训练入口
- `run_finetune.bash`：批量实验脚本
- `data/`：数据文件与预处理脚本
- `Exp6_WeaktoStrong/`：批量运行日志
- `output/`：训练输出与检查点

`Baselines/` 当前包含：

- `Neg_samples_DNS+`
- `Neg_samples_gnno`
- `Neg_samples_posmix`
- `Neg_samples_srns`
- `Neg_samples_two_pass`

## ⚙️ 环境依赖

仓库中暂未提供固定版本的 `requirements.txt`，根据代码导入情况，建议至少准备以下环境：

- Python 3.9+
- PyTorch
- NumPy / SciPy / pandas / scikit-learn
- tqdm / texttable / openpyxl
- transformers
- `mamba-ssm`
- recbole

如果你只运行部分模型，实际依赖可能少于上面列表；如果启用 `Mamba4Rec`、部分 baseline 或扩展 backbone，则需要额外确保对应依赖可用。

## 🚀 快速开始

### 1. 进入主代码目录

```bash
cd MDCNS_Code
```

### 2. 准备数据

主程序默认从 `./data/` 读取数据，命名格式为：

```text
<Dataset>_train.txt
<Dataset>_val.txt
<Dataset>_test.txt
```

例如：

- `Beauty_train.txt`
- `Beauty_val.txt`
- `Beauty_test.txt`

当前仓库中已包含 `Beauty` 的处理后划分文件，其它数据集可以按同样命名规则放入 `MDCNS_Code/data/`。

### 3. 运行批量实验

```bash
bash run_finetune.bash
```

注意：

- 实际可执行脚本位于 `MDCNS_Code/run_finetune.bash`
- 脚本中默认使用多卡并发：`AVAILABLE_GPUS=(1 2 3 0)`
- 运行前建议先按你的机器配置修改 GPU 列表、并发数、数据集列表与超参数范围

## 🧪 单次运行示例

如果你想手动跑一个实验，可以直接执行：

```bash
cd MDCNS_Code
python3 -u main.py \
  --data_name Beauty \
  --backbone SASRec \
  --backbone2 GRU4Rec \
  --hidden_size 64 \
  --loss_type BCE \
  --neg_sampler DNS \
  --N 100 \
  --K_hns 1 \
  --alpha 1.0 \
  --beta 1.0 \
  --d_lambda 3.5 \
  --dws_beta 1.0 \
  --temperature 0.0 \
  --kd_temperature 1.0 \
  --kd_gamma 0.01
```

## 📊 输出说明

运行过程中常见输出位置：

- 日志：`MDCNS_Code/Exp6_WeaktoStrong/`
- 模型输出与检查点：`MDCNS_Code/output/`

批量脚本会自动把每组超参数的 stdout/stderr 重定向到日志文件中，便于后续筛选结果。

## 🗂️ 数据与预处理

数据相关文件位于 `MDCNS_Code/data/`，其中包括：

- 已处理好的 `Beauty` 训练/验证/测试文件
- `data_process.py` 与若干 notebook，用于数据处理与分析

主程序读取逻辑写在 `MDCNS_Code/main.py`，默认会拼接为：

```text
./data/<Dataset>_train.txt
./data/<Dataset>_val.txt
./data/<Dataset>_test.txt
```

## 🧩 Baseline 说明

`Baselines/` 中的每个子目录都是相对独立的实验实现，一般包含：

- `run_finetune.bash`
- `run_finetune_full.py`
- `models.py`
- `modules.py`
- `trainers.py`

如果你要复现实验对比，建议进入对应 baseline 目录后单独运行其脚本。

## 📄 论文材料

- 论文/材料文件：`SR_CR.pdf`

如果后续补充正式论文链接、BibTeX 或实验表格，推荐继续放在仓库根目录并在本 README 中统一索引。
