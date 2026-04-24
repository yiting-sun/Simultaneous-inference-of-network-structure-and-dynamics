#!/bin/bash
# Lorenz simultaneous inference (structure + dynamics) on dolphins-like 46-node subgraphs
# (component-R² validation: self_x/y/z + Aij * G_true vs soft_w * msg_pred).
#
# Output: Lorenz_dyn/results_simul/

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_simul_Lorenz.py"

BASE_OUTDIR="${SCRIPT_DIR}/results_simul"
LOGDIR="${BASE_OUTDIR}/logs"
mkdir -p "${LOGDIR}"

GPUS=(0 1 2 3)
MAX_PER_GPU=10

DATA_SEED=1
SEEDS=(1 2 3 4 5)
HIDDEN=50
EPOCHS=2500
LAM=0.1
TAU_DECAY=0.999
BATCH_RATIO=0.05
USE_EARLY_STOP=false

# Format: "N  start  step  end"
# Simul requires more training samples than knstruc, so ranges are wider.
NODE_SPECS=(
    "5    5   2   80"
    "10   5   2   40"
    "15   3   2   40"
    "20   2   2   50"
    "25   2   2   50"
    "30   3   2   72"
    "35   3   2   72"
    "40   2   2   82"
    "45   2   3   102"
    "50   2   5   102"
    "55   2   5   102"
    "58   2   5   102"
)

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

for spec in "${NODE_SPECS[@]}"; do
    read -r NUM_NODES_KEEP SEQ_START SEQ_STEP SEQ_END <<< "$spec"
    TRAIN_SAMPLES=($(seq ${SEQ_START} ${SEQ_STEP} ${SEQ_END}))

    for n_train in "${TRAIN_SAMPLES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            RUN_NAME="Nodes${NUM_NODES_KEEP}_Ntrain${n_train}_hidden${HIDDEN}_seed${seed}"
            OUTDIR="${BASE_OUTDIR}/${RUN_NAME}"

            if [ -f "${OUTDIR}/results.json" ]; then
                SKIPPED=$((SKIPPED + 1))
                continue
            fi

            GPU_ID=$(wait_for_slot)
            TOTAL=$((TOTAL + 1))

            echo "[GPU ${GPU_ID}] #${TOTAL} N=${NUM_NODES_KEEP} ${RUN_NAME}"
            python "${TRAIN_SCRIPT}" \
                --outdir "${OUTDIR}/" \
                --num_nodes_keep "${NUM_NODES_KEEP}" \
                --num_train_samples "${n_train}" \
                --data_seed "${DATA_SEED}" \
                --seed "${seed}" \
                --hidden "${HIDDEN}" \
                --epochs "${EPOCHS}" \
                --lam "${LAM}" \
                --tau_update "${TAU_DECAY}" \
                --batch_ratio "${BATCH_RATIO}" \
                --lr 1e-3 \
                --use_early_stop "${USE_EARLY_STOP}" \
                --device_id "${GPU_ID}" \
                > "${LOGDIR}/${RUN_NAME}.log" 2>&1 &

            gpu_pids[$GPU_ID]="${gpu_pids[$GPU_ID]} $!"
        done
    done
    echo "--- N=${NUM_NODES_KEEP}: all jobs submitted ---"
done

echo ""
echo "All ${TOTAL} Lorenz simul jobs launched (${SKIPPED} skipped). Waiting..."
wait
echo "All Lorenz simul experiments completed."
