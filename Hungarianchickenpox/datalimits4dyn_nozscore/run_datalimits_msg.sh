#!/bin/bash
# Run data-limits sweep for fixed-adj message-feature ablation.
# Sweeps --num_train_samples to study how much training data is required for
# dynamics learning. Features are built on the FULL series first, then the
# training set is uniformly downsampled (val/test stay at the full size for
# fair comparison across runs).
#
# Distributes jobs across 4 GPUs (cuda:0-3), max 5 concurrent per GPU.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_fixed_adj_ablation_msg_datalimits.py"
# Number of nodes in the graph (chickenpox Hungary = 20 counties). The output
# directory name records this so multi-node-count runs don't collide.
NUM_NODES=20
BASE_OUTDIR="${SCRIPT_DIR}/runs_datalimits_msg_nodes${NUM_NODES}"
LOGDIR="${BASE_OUTDIR}/logs"
mkdir -p "${LOGDIR}"

GPUS=(0 1 2 3)
MAX_PER_GPU=5

SEEDS=(1 2 3 4 5)
MSG_CONFIGS=(rate_x)
ADJ_TYPES=(runs3thr0p6)
ZSCORES=(false)

# Training-data sizes to sweep. With 60/20/20 split on ~248 feature rows the
# training set has ~148 rows; adjust if your series length changes.
# NUM_TRAIN_SAMPLES=(2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40)
NUM_TRAIN_SAMPLES=($(seq 2 2 40))

# Batch size as a fraction of the (subsampled) training set, so the number of
# optimizer steps per epoch stays roughly constant across different
# --num_train_samples.
BATCH_RATIO=0.1

declare -A gpu_count
declare -A gpu_pids
for g in "${GPUS[@]}"; do
    gpu_count[$g]=0
    gpu_pids[$g]=""
done

wait_for_slot() {
    while true; do
        for g in "${GPUS[@]}"; do
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

TOTAL=0
SKIPPED=0

for adj_type in "${ADJ_TYPES[@]}"; do
    for msg in "${MSG_CONFIGS[@]}"; do
        for zscore in "${ZSCORES[@]}"; do
            if [ "$zscore" = "true" ]; then
                zscore_tag="zscore"
            else
                zscore_tag="nozscore"
            fi
            for n_train in "${NUM_TRAIN_SAMPLES[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    RUN_NAME="run${seed}_${adj_type}_${msg}_${zscore_tag}_gaussian_n${n_train}"
                    OUTDIR="${BASE_OUTDIR}/${RUN_NAME}"

                    if [ -f "${OUTDIR}/results.json" ]; then
                        echo "SKIP: ${RUN_NAME}"
                        SKIPPED=$((SKIPPED + 1))
                        continue
                    fi

                    GPU_ID=$(wait_for_slot)
                    TOTAL=$((TOTAL + 1))

                    echo "[GPU ${GPU_ID}] #${TOTAL} START: ${RUN_NAME}"
                    python "${TRAIN_SCRIPT}" \
                        --outdir "${OUTDIR}" \
                        --adj_type "${adj_type}" \
                        --msg_config "${msg}" \
                        --zscore "${zscore}" \
                        --smooth gaussian \
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
                        > "${LOGDIR}/${RUN_NAME}.log" 2>&1 &

                    gpu_pids[$GPU_ID]="${gpu_pids[$GPU_ID]} $!"
                done
            done
        done
    done
done

echo ""
echo "All ${TOTAL} jobs launched (${SKIPPED} skipped). Waiting for completion..."
wait
echo "All data-limits ablation experiments completed."
