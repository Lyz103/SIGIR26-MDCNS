#!/usr/bin/env bash

export LD_LIBRARY_PATH=""
set -euo pipefail
IFS=$'\n\t'

# ———————— 可配置项 ————————
# 模型与数据集
MODELS=("Mamba4Rec" "GRU4Rec" "LightSANs" "FMLPRecModel" "Linrec")
DATASETS=("Toys_and_Games" "Beauty" "Yelp" "Video_Games" "Sports_and_Outdoors")
# 你可以取消上面一行，使用下面这行来调试单个数据集
# DATASETS=("Beauty")

# 单个模型调试用
# MODELS=("SASRec")
DATASETS=("Beauty")
DATASETS=("Beauty" "Toys_and_Games" "Yelp" "Sports_and_Outdoors")
# 单个模型调试用
MODELS=("SASRec")
START_EPOCHS=(0)

# 超参数
NEG_SAMPLER="DNS"
CL_TYPE="Radical"
HIDDEN_SIZE=64
N=(500)  # ← 将被逐个扫描
# N=(100)

# 并发控制
JOBS_PER_GPU=1
AVAILABLE_GPUS=(0 1 2 3)  # ← 修改为你实际可用的 GPU ID 列表
NUM_GPUS=${#AVAILABLE_GPUS[@]}

if (( NUM_GPUS == 0 )); then
  echo "Error: 请在脚本中手动设置 AVAILABLE_GPUS，至少一个 GPU ID！" >&2
  exit 1
fi

TOTAL_SLOTS=$(( NUM_GPUS * JOBS_PER_GPU ))

echo "使用手动指定的 GPU 列表: ${AVAILABLE_GPUS[*]}"
echo "每 GPU 并发: $JOBS_PER_GPU，总并发: $TOTAL_SLOTS"

# 输出目录
OUTDIR="res2"
mkdir -p "$OUTDIR"

# ———————— 并发控制函数 ————————
running_jobs() {
  jobs -rp | wc -l
}

wait_for_slot() {
  while (( $(running_jobs) >= TOTAL_SLOTS )); do
    wait -n || :
  done
}

# 轮询分配 GPU
GPU_INDEX=0

# ———————— 信号处理：清理所有子进程 ————————
trap_cleanup() {
  echo ""
  echo "[$(date '+%H:%M:%S')] 检测到中断信号 (Ctrl+C)，正在终止所有子进程..."
  jobs -p | xargs kill -TERM 2>/dev/null || true
  wait || true
  echo "所有子进程已终止。"
  exit 130  # SIGINT 标准退出码
}

# 设置信号陷阱（捕获 Ctrl+C）
trap trap_cleanup SIGINT SIGTERM

# ———————— 主循环 ————————
for start_epoch in "${START_EPOCHS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
      for n_val in "${N[@]}"; do  # ← 新增：遍历 N 的每个值
        wait_for_slot

        gpu="${AVAILABLE_GPUS[GPU_INDEX]}"
        GPU_INDEX=$(((GPU_INDEX + 1) % NUM_GPUS))

        logfile="$OUTDIR/${ds}-${model}_epoch${start_epoch}_N${n_val}.log"
        echo "[$(date '+%H:%M:%S')] 启动 $model@$ds epoch=$start_epoch N=$n_val → GPU $gpu"

        # 在后台运行任务
        CUDA_VISIBLE_DEVICES="$gpu" python3 -u run_finetune_full.py \
          --data_name="$ds" \
          --ckp=0 \
          --hidden_size="$HIDDEN_SIZE" \
          --start_epoch="$start_epoch" \
          --N="$n_val" \
          --neg_sampler="$NEG_SAMPLER" \
          --CL_type="$CL_TYPE" \
          --backbone="$model" \
        > "$logfile" 2>&1 &

      done
    done
  done
done

# 等待所有子进程完成
wait
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 全部任务完成！"