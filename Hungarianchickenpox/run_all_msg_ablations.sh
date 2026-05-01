#!/bin/bash
# Run all message-feature ablation experiments (simultaneous inference)
# Distributes jobs across 4 GPUs (cuda:0-3), max 5 concurrent per GPU.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_ablation_msg.py"
BASE_OUTDIR="${SCRIPT_DIR}/runs_msg_ablation_tau0.99"
LOGDIR="${BASE_OUTDIR}/logs"
mkdir -p "${LOGDIR}"

GPUS=(0 1 2 3)
MAX_PER_GPU=5

SEEDS=(1 2 3)
MSG_CONFIGS=(rate_x)
LAMS=(0.001)
ZSCORES=(false)

# Track how many jobs are running on each GPU
declare -A gpu_count
for g in "${GPUS[@]}"; do gpu_count[$g]=0; done

# Wait until a GPU has a free slot, return its index
wait_for_slot() {
    while true; do
        for g in "${GPUS[@]}"; do
            # Recount: number of alive background children on this GPU
            local cnt=0
            for pid in ${gpu_pids[$g]}; do
                if kill -0 "$pid" 2>/dev/null; then
                    cnt=$((cnt + 1))
                fi
            done
            gpu_count[$g]=$cnt
            if [ "$cnt" -lt "$MAX_PER_GPU" ]; then
                echo "$g"
                return
            fi
        done
        sleep 2
    done
}

declare -A gpu_pids  # space-separated PIDs per GPU
for g in "${GPUS[@]}"; do gpu_pids[$g]=""; done

TOTAL=0
SKIPPED=0

for msg in "${MSG_CONFIGS[@]}"; do
    for lam in "${LAMS[@]}"; do
        for zscore in "${ZSCORES[@]}"; do
            if [ "$zscore" = "true" ]; then
                zscore_tag="zscore"
            else
                zscore_tag="nozscore"
            fi
            for seed in "${SEEDS[@]}"; do
                RUN_NAME="run${seed}_${msg}_lam${lam}_${zscore_tag}_gaussian_tau0.99"
                OUTDIR="${BASE_OUTDIR}/${RUN_NAME}"

                if [ -f "${OUTDIR}/final_model.pt" ]; then
                    echo "SKIP: ${RUN_NAME}"
                    SKIPPED=$((SKIPPED + 1))
                    continue
                fi

                # Get a GPU with a free slot
                GPU_ID=$(wait_for_slot)
                TOTAL=$((TOTAL + 1))

                echo "[GPU ${GPU_ID}] #${TOTAL} START: ${RUN_NAME}"
                python "${TRAIN_SCRIPT}" \
                    --outdir "${OUTDIR}" \
                    --msg_config "${msg}" \
                    --zscore "${zscore}" \
                    --smooth gaussian \
                    --tau 1.0 \
                    --tau_decay 0.99 \
                    --lam "${lam}" \
                    --hidden 100 \
                    --epochs 2000 \
                    --batch 64 \
                    --lr 1e-3 \
                    --early_stop_patience 40 \
                    --use_seed true \
                    --seed "${seed}" \
                    --device "cuda:${GPU_ID}" \
                    --train_ratio 0.6 \
                    --val_ratio 0.2 \
                    --test_ratio 0.2 \
                    > "${LOGDIR}/${RUN_NAME}.log" 2>&1 &

                gpu_pids[$GPU_ID]="${gpu_pids[$GPU_ID]} $!"
            done
        done
    done
done

echo ""
echo "All ${TOTAL} jobs launched (${SKIPPED} skipped). Waiting for completion..."
wait
echo "All message-feature ablation experiments completed."
