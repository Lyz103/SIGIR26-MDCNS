#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# ———————— 可配置项 ————————
# 模型与数据集
MODELS=("Mamba4Rec" "GRU4Rec" "LightSANs" "FMLPRecModel" "Linrec")
DATASETS=("Toys_and_Games" "Beauty" "Yelp" "Video_Games" "Sports_and_Outdoors")
DATASETS=("Beauty")
MODELS=("SASRec")
DATASETS=("LastFM")
# start_epoch 从 0 到 70
START_EPOCHS=(0)

# 负采样器、对比学习类型等超参
NEG_SAMPLER="DNS"
CL_TYPE="Radical"
HIDDEN_SIZE=64
N=100
M=(1 2 3 4 5 6 7 8 9 10 15 20 25 30 35 40 45 50 100)
# M=(20)

# 每个 GPU 最多运行多少作业
JOBS_PER_GPU=2

# 手动指定可用 GPU 列表（自行修改）
AVAILABLE_GPUS=(4)
NUM_GPUS=${#AVAILABLE_GPUS[@]}
if (( NUM_GPUS == 0 )); then
  echo "Error: 请在脚本中手动设置 AVAILABLE_GPUS，至少一个 GPU ID！"
  exit 1
fi
TOTAL_SLOTS=$(( NUM_GPUS * JOBS_PER_GPU ))

echo "使用手动指定的 GPU 列表: ${AVAILABLE_GPUS[*]}"
echo "每 GPU 并发: $JOBS_PER_GPU，总并发: $TOTAL_SLOTS"

# 输出目录
OUTDIR="res"
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

# 轮询分配 GPU：循环轮询可用 slots，配合 round-robin
GPU_INDEX=0

# ———————— 主循环 ————————
for start_epoch in "${START_EPOCHS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
      for m in "${M[@]}"; do  # 新增扫描 M 的循环

        wait_for_slot

        gpu="${AVAILABLE_GPUS[GPU_INDEX]}"
        GPU_INDEX=$(((GPU_INDEX + 1) % NUM_GPUS))

        logfile="$OUTDIR/${ds}-${model}_epoch${start_epoch}_M${m}.log"
        echo "[$(date '+%H:%M:%S')] 启动 $model@$ds epoch=$start_epoch M=$m → GPU $gpu"

        CUDA_VISIBLE_DEVICES="$gpu" python3 -u run_finetune_full.py \
          --data_name="$ds" \
          --ckp=0 \
          --hidden_size="$HIDDEN_SIZE" \
          --start_epoch="$start_epoch" \
          --N="$N" \
          --M="$m" \
          --neg_sampler="$NEG_SAMPLER" \
          --CL_type="$CL_TYPE" \
          --backbone="$model" \
        > "$logfile" 2>&1 &

      done
    done
  done
done

# 等待所有子进程结束
wait
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 全部任务完成！"
