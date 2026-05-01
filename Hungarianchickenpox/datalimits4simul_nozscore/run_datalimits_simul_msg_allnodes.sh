#!/bin/bash
# Simultaneous-inference msg-feature ablation — sweep ALL node sizes.
#
# Each NUM_NODES_KEEP has its own n_train range (tailored so that the
# largest total_samples ≈ N × max_n_train stays reasonable).
#
# Output: runs_datalimits_simul_msg_nodes${NUM_NODES_KEEP}/
# Distributes jobs across 4 GPUs (cuda:0-3), max 12 concurrent per GPU.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_ablation_msg_subnodes.py"

GPUS=(0 1 2 3)
MAX_PER_GPU=5

SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)
MSG_CONFIGS=(rate_x)
LAMS=(0.001)
TAU=1.0
TAU_DECAY=0.99
BATCH_RATIO=0.1

declare -A gpu_pids
for g in "${GPUS[@]}"; do gpu_pids[$g]=""; done

wait_for_slot() {
    while true; do
        for g in "${GPUS[@]}"; do
            local cnt=0
            for pid in ${gpu_pids[$g]}; do
                if kill -0 "$pid" 2>/dev/null; then
                    cnt=$((cnt + 1))
                fi
            done
            if [ "$cnt" -lt "$MAX_PER_GPU" ]; then
                echo "$g"
                return
            fi
        done
        sleep 2
    done
}

TOTAL=0
SKIPPED=0

# ---- Per-node-size sweep definitions --------------------------------
# Format: NUM_NODES_KEEP  DIR_SUFFIX  SEQ_START SEQ_STEP SEQ_END
#   n_train values = $(seq SEQ_START SEQ_STEP SEQ_END)
NODE_SPECS=(
    # "3   runs_datalimits_simul_msg_nodes3   2 8 160"
    "5   runs_datalimits_simul_msg_nodes5   2 8 160"
    "7   runs_datalimits_simul_msg_nodes7   2 6 120"
    "9   runs_datalimits_simul_msg_nodes9   2 5 100"
    "11  runs_datalimits_simul_msg_nodes11  2 4 80"
    "13  runs_datalimits_simul_msg_nodes13  2 4 70"
    "15  runs_datalimits_simul_msg_nodes15  2 3 54"
    "17  runs_datalimits_simul_msg_nodes17  2 2 50"
    "19  runs_datalimits_simul_msg_nodes19  2 2 50"
    "20  runs_datalimits_simul_msg_nodes20  2 2 41"
)

for spec in "${NODE_SPECS[@]}"; do
    read -r NUM_NODES_KEEP DIR_SUFFIX SEQ_START SEQ_STEP SEQ_END <<< "$spec"

    BASE_OUTDIR="${SCRIPT_DIR}/${DIR_SUFFIX}"
    LOGDIR="${BASE_OUTDIR}/logs"
    mkdir -p "${LOGDIR}"

    NUM_TRAIN_SAMPLES=($(seq ${SEQ_START} ${SEQ_STEP} ${SEQ_END}))

    for msg in "${MSG_CONFIGS[@]}"; do
        for lam in "${LAMS[@]}"; do
            for n_train in "${NUM_TRAIN_SAMPLES[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    RUN_NAME="run${seed}_${msg}_lam${lam}_nozscore_gaussian_tau${TAU_DECAY}_n${n_train}"
                    OUTDIR="${BASE_OUTDIR}/${RUN_NAME}"

                    if [ -f "${OUTDIR}/final_model.pt" ]; then
                        SKIPPED=$((SKIPPED + 1))
                        continue
                    fi

                    GPU_ID=$(wait_for_slot)
                    TOTAL=$((TOTAL + 1))

                    echo "[GPU ${GPU_ID}] #${TOTAL} N=${NUM_NODES_KEEP} ${RUN_NAME}"
                    python "${TRAIN_SCRIPT}" \
                        --outdir "${OUTDIR}" \
                        --msg_config "${msg}" \
                        --zscore false \
                        --smooth gaussian \
                        --tau "${TAU}" \
                        --tau_decay "${TAU_DECAY}" \
                        --lam "${lam}" \
                        --hidden 100 \
                        --epochs 2000 \
                        --batch_ratio "${BATCH_RATIO}" \
                        --lr 1e-3 \
                        --early_stop_patience 40 \
                        --use_seed true \
                        --seed "${seed}" \
                        --device "cuda:${GPU_ID}" \
                        --train_ratio 0.6 \
                        --val_ratio 0.2 \
                        --test_ratio 0.2 \
                        --num_train_samples "${n_train}" \
                        --num_nodes_keep "${NUM_NODES_KEEP}" \
                        > "${LOGDIR}/${RUN_NAME}.log" 2>&1 &

                    gpu_pids[$GPU_ID]="${gpu_pids[$GPU_ID]} $!"
                done
            done
        done
    done
    echo "--- N=${NUM_NODES_KEEP}: all jobs submitted ---"
done

echo ""
echo "All ${TOTAL} jobs launched (${SKIPPED} skipped). Waiting for completion..."
wait
echo "All node-size sweeps completed."
