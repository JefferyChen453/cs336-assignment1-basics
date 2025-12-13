#!/bin/bash
set -e

#############################################
# 固定配置文件路径（根据你的项目结构可调整）
#############################################
CONFIG="cs336_basics/config/train_tinystories.yaml"

#############################################
# 1. 学习率粗搜索（log-scale）
#############################################
LR_COARSE=(
    1e-5
    5e-5
    1e-4
    5e-4
    1e-3
    5e-3
    1e-2
)

echo "===== LR COARSE SWEEP START ====="
for LR in "${LR_COARSE[@]}"; do
    EXP_NAME="lr_coarse_${LR}"
    echo "[RUN] coarse LR sweep, lr=${LR}"

    uv run -m cs336_basics.transformer.train.main \
        --config ${CONFIG} \
        --optim.lr ${LR} \
        --trainer.total_steps 5000 \
        --trainer.save_freq 999999 \
        --trainer.experiment_name ${EXP_NAME} \
        --debug.debug_mode false
done
echo "===== LR COARSE SWEEP END ====="


############################################
2. 学习率细搜索（你可以根据 coarse 结果调区间）
############################################
假设 coarse sweep 结果显示稳定区大约是 [1e-4, 1e-3]
LR_FINE=(
    6e-3
    7e-3
    8e-3
    9e-3
    2e-3
    3e-3
    4e-3
)

echo "===== LR FINE SWEEP START ====="
for LR in "${LR_FINE[@]}"; do
    EXP_NAME="lr_fine_${LR}"

    echo "[RUN] fine LR sweep, lr=${LR}"

    uv run -m cs336_basics.transformer.train.main \
        --config ${CONFIG} \
        --optim.lr ${LR} \
        --trainer.total_steps 5000 \
        --trainer.save_freq 999999 \
        --trainer.experiment_name ${EXP_NAME}
done
echo "===== LR FINE SWEEP END ====="


#############################################
# 3. Batch size 实验 + LR scaling
#############################################
BATCH_LIST=(1 2 4 8 16 32 64 128 256)

# baseline lr（你当前配置文件里的 lr）
BASE_LR=5e-3

echo "===== BATCH SIZE EXPERIMENT START ====="

for BS in "${BATCH_LIST[@]}"; do
    # scaling 因子
    SCALE=$(echo "$BS / 128" | bc -l)

    # linear scaling: lr = BASE * k
    LR_LINEAR=$(python3 - <<EOF
k=${SCALE}
base=${BASE_LR}
print(base * k)
EOF
)

    # sqrt scaling: lr = BASE * sqrt(k)
    LR_SQRT=$(python3 - <<EOF
import math
k=${SCALE}
base=${BASE_LR}
print(base * math.sqrt(k))
EOF
)

    echo "--- batch size = ${BS} ---"
    echo "  linear lr  = $LR_LINEAR"
    echo "  sqrt lr    = $LR_SQRT"

    ###################################################
    # Run 1: Linear scaling
    ###################################################
    uv run -m cs336_basics.transformer.train.main \
        --config ${CONFIG} \
        --data.train_batch_size ${BS} \
        --optim.lr ${LR_LINEAR} \
        --trainer.total_steps 10000 \
        --trainer.save_freq 999999 \
        --trainer.experiment_name "bs_${BS}_linear_lr_${LR_LINEAR}"

    ###################################################
    # Run 2: Sqrt scaling
    ###################################################
    uv run -m cs336_basics.transformer.train.main \
        --config ${CONFIG} \
        --data.train_batch_size ${BS} \
        --optim.lr ${LR_SQRT} \
        --trainer.total_steps 10000 \
        --trainer.save_freq 999999 \
        --trainer.experiment_name "bs_${BS}_sqrt_lr_${LR_SQRT}"

    ###################################################
    # Run 3: No scaling（baseline lr）
    ###################################################
    uv run -m cs336_basics.transformer.train.main \
        --config ${CONFIG} \
        --data.train_batch_size ${BS} \
        --optim.lr ${BASE_LR} \
        --trainer.total_steps 10000 \
        --trainer.save_freq 999999 \
        --trainer.experiment_name "bs_${BS}_noscale_lr_${BASE_LR}"
done

echo "===== BATCH SIZE EXPERIMENT END ====="


#############################################
# FINISH
#############################################
echo "===== ALL EXPERIMENTS COMPLETED ====="



uv run -m cs336_basics.transformer.train.main \
    --config /workspace/cs336-assignment1-basics/cs336_basics/config/train_tinystories.yaml \
    --data.train_batch_size 128 \
    --optim.lr 5e-3 \
    --trainer.total_steps 10000 \
    --trainer.save_freq 999999 \
    --trainer.project_name "tinystories-train" \
    --trainer.experiment_name "final"

uv run -m cs336_basics.transformer.train.main \
    --config /workspace/cs336-assignment1-basics/cs336_basics/config/train_owt.yaml \
    --data.train_batch_size 128 \
    --optim.lr 5e-3 \
    --trainer.total_steps 10000 \
    --trainer.save_freq 999999 \
    --trainer.project_name "train-owt" \
    --trainer.experiment_name "final"
