#!/usr/bin/env bash
set -euo pipefail
# 设置字段分隔符，防止在处理数组时出现意外行为
IFS=$'\n\t'

# ———————— 可配置项 (DWS + KD 版本) ————————
# 基础模型列表 (脚本会自动组合成不重复的模型对)
MODELS=("SASRec" "GRU4Rec")

# 数据集列表
DATASETS=("Beauty" "Toys_and_Games" "Sports_and_Outdoors" "LastFM") # 
# DATASETS=("KuaiRand") # 

# DWS 框架的超参数
# ==========================================================
ALPHA_VALUES=(1.0)
BETA_VALUES=(1.0)
D_LAMBDA_VALUES=(3.5 3.0 2.5 0.3 0.5 1.0 1.5 2.0)
DWS_BETA_VALUES=(1.0)
TEMPERATURE_VALUES=(0.0)

# ▼▼▼ KD 协同蒸馏超参数 (新增) ▼▼▼
# (推荐的 T 搜索范围, T=1.0 会导致暗知识丢失)
KD_TEMPERATURE_VALUES=(0.7 1.0 2.0 5.0 10.0)
# (推荐的 gamma 搜索范围, 注意 T^2 会放大该值. 0.0 表示关闭 KD)
KD_GAMMA_VALUES=(0.01)
# ==========================================================

# 困难负采样池大小 K
K_HNS_VALUES=(1)

# 候选负采样池大小 (用于生成负样本候选)
NEG_SAMPLER="DNS"
N_VALUES=(100) 
HIDDEN_SIZE=64
LOSS_TYPE="BCE" # "BPR" 或 "BCE"

# ———————— 并发与 GPU 配置 ————————
# 限制：每个 GPU 并行运行的最大任务数
JOBS_PER_GPU=2

# 可用的 GPU ID 列表
AVAILABLE_GPUS=(1 2 3 0)

# 检查 GPU 配置
NUM_GPUS=${#AVAILABLE_GPUS[@]}
if (( NUM_GPUS == 0 )); then
  echo "Error: 请在脚本中手动设置 AVAILABLE_GPUS，至少一个 GPU ID！"
  exit 1
fi

echo "使用 GPU 列表: ${AVAILABLE_GPUS[*]}"
echo "每 GPU 最大并发数: $JOBS_PER_GPU"

# ———————— 状态跟踪和清理 ————————
OUTDIR="Exp6_WeaktoStrong" # <-- 修改了输出目录名
mkdir -p "$OUTDIR"

# 关联数组：用于跟踪 PID 及其对应的 GPU ID
# 格式: [PID]=GPU_ID
declare -A PID_TO_GPU_MAP

# 退出/中断处理函数
cleanup() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 捕获到中断信号，正在清理所有子进程..."
  # pkill -P $$ 会杀掉所有由当前脚本（父进程ID为$$）启动的子进程
  # 必须使用 kill 命令，因为 wait -n 依赖于后台 job 而非 PID 数组
  for pid in "${!PID_TO_GPU_MAP[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
          echo "  - 终止 PID $pid (GPU ${PID_TO_GPU_MAP[$pid]})"
          kill "$pid" || true
      fi
  done
  exit 1
}

trap cleanup SIGINT SIGTERM SIGQUIT

# ———————— 资源管理函数 ————————

# 功能: 计算指定 GPU 上运行中的任务数量，并清理已完成的任务 PID
# 参数: $1 - 目标 GPU ID
# 返回: 运行中的任务数量
count_gpu_jobs() {
    local target_gpu="$1"
    local count=0
    local pids_to_remove=()
    
    # 遍历所有正在跟踪的 PID
    for pid in "${!PID_TO_GPU_MAP[@]}"; do
        local assigned_gpu="${PID_TO_GPU_MAP[$pid]}"
        
        # 1. 检查进程是否仍在运行 (kill -0 不发送信号，只检查权限和存在性)
        if kill -0 "$pid" 2>/dev/null; then
            # 仍在运行, 检查是否是目标 GPU 的任务
            if [[ "$assigned_gpu" == "$target_gpu" ]]; then
                ((count++))
            fi
        else
            # 进程已完成，标记删除
            pids_to_remove+=("$pid")
        fi
    done
    
    # 清理已完成的 PID
    for pid in "${pids_to_remove[@]}"; do
        unset 'PID_TO_GPU_MAP[pid]'
    done
    
    echo "$count"
}

# 功能: 阻塞等待任意一个子进程完成
wait_for_any_slot() {
    echo "--- 所有 GPU 已饱和 (每张卡 $JOBS_PER_GPU 个任务). 阻塞等待槽位释放... ---"
    # wait -n 等待任意一个后台 job 完成
    # 这里的 job 必须是使用 '&' 启动的进程
    wait -n || :
}

# ———————— 主循环 ————————
# GPU 轮询索引：从哪个 GPU 开始搜索可用槽位
GPU_INDEX=0

for ds in "${DATASETS[@]}"; do
for N in "${N_VALUES[@]}"; do 
for alpha in "${ALPHA_VALUES[@]}"; do
for beta in "${BETA_VALUES[@]}"; do
for d_lambda in "${D_LAMBDA_VALUES[@]}"; do
for dws_beta in "${DWS_BETA_VALUES[@]}"; do
for temp in "${TEMPERATURE_VALUES[@]}"; do
for k_hns in "${K_HNS_VALUES[@]}"; do
for kd_temp in "${KD_TEMPERATURE_VALUES[@]}"; do # <-- 新增 KD 温度循环
for kd_gamma in "${KD_GAMMA_VALUES[@]}"; do # <-- 新增 KD Gamma 循环

  # 遍历模型对
  for (( i=0; i<${#MODELS[@]}; i++ )); do
    for (( j=i+1; j<${#MODELS[@]}; j++ )); do
      model1="${MODELS[i]}"
      model2="${MODELS[j]}"

      # ———————— 资源分配循环 ————————
      while true; do
        found_slot=0
        start_index=$GPU_INDEX
        
        # 从当前的 GPU_INDEX 开始，循环检查所有 GPU
        for (( k=0; k<NUM_GPUS; k++ )); do
          gpu_check_index=$(((start_index + k) % NUM_GPUS))
          current_gpu="${AVAILABLE_GPUS[gpu_check_index]}"
          
          # 获取当前 GPU 上的运行任务数 (同时会清理已完成的 PID)
          current_jobs=$(count_gpu_jobs "$current_gpu")
          
          if (( current_jobs < JOBS_PER_GPU )); then
            # 找到可用槽位
            gpu="$current_gpu"
            # 更新下一轮的起始轮询索引 (保持轮询公平性)
            GPU_INDEX=$(((gpu_check_index + 1) % NUM_GPUS)) 
            found_slot=1
            break
          fi
        done
        
        if (( found_slot == 1 )); then
          break # 找到槽位，退出等待循环，准备启动任务
        else
          # 所有 GPU 都已饱和，阻塞等待任一任务完成
          wait_for_any_slot
          # 循环将再次启动，并重新检查 GPU 上的任务数
        fi
      done
      # ———————— 资源分配结束 ————————
      
      # <-- 修改点：日志文件名中加入 kd_temp 和 kd_gamma 的值 -->
      logfile="$OUTDIR/${ds}-${model1}-${model2}_a${alpha}_b${beta}_l${d_lambda}_dwsb${dws_beta}_t${temp}_k${k_hns}_N${N}_kdt${kd_temp}_kdg${kd_gamma}.log"
      
      # <-- 修改点：echo 中加入 kd_T 和 kd_G 的值 -->
      echo "[$(date '+%H:%M:%S')] 启动 ${model1}-${model2}@${ds} (a=${alpha},b=${beta},λ=${d_lambda},dws_b=${dws_beta},τ=${temp},K=${k_hns},N=${N},kd_T=${kd_temp},kd_G=${kd_gamma}) → GPU $gpu (当前任务数: $(count_gpu_jobs "$gpu"))"
      
      # 使用新的DWS参数启动 main.py
      # 注意：-u 参数确保 python 输出不会被缓冲
      CUDA_VISIBLE_DEVICES="$gpu" python3 -u main.py \
        --data_name="$ds" \
        --backbone="$model1" \
        --backbone2="$model2" \
        --hidden_size="$HIDDEN_SIZE" \
        --loss_type="$LOSS_TYPE" \
        --neg_sampler="$NEG_SAMPLER" \
        --N="$N" \
        --K_hns="$k_hns" \
        --alpha="$alpha" \
        --beta="$beta" \
        --d_lambda="$d_lambda" \
        --dws_beta="$dws_beta" \
        --temperature="$temp" \
        --kd_temperature="$kd_temp" \
        --kd_gamma="$kd_gamma" \
        > "$logfile" 2>&1 & # 将标准输出和标准错误都重定向到日志文件
      
      # 记录子进程 PID 及其分配的 GPU ID
      PID_TO_GPU_MAP[$!]=${gpu}
      echo "  - 日志: $logfile"
    done
  done
  
done # <-- 新增: 对应 kd_gamma 循环
done # <-- 新增: 对应 kd_temp 循环
done
done
done
done
done
done
done
done 

# 再次执行一次清理，确保 PID_TO_GPU_MAP 准确
count_gpu_jobs -1 # 随便传一个 GPU ID，目的是触发 cleanup 逻辑

# 等待所有后台任务完成
echo "----------------------------------------------------------------"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 所有实验任务已启动，等待全部完成..."
# wait -n 循环等待所有任务完成
while (( ${#PID_TO_GPU_MAP[@]} > 0 )); do
    wait_for_any_slot
    # 每次 wait -n 完成后，再次调用 count_gpu_jobs 触发清理
    count_gpu_jobs -1
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 全部任务完成！"