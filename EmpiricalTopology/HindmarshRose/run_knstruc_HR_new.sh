#!/bin/bash
# Known-structure dynamics inference on degree-matched HR Celegans subgraphs
# using COMPONENT-R² validation (self_x/y/z + message).
#
# Output: HR_dyn/results_knstruc_new/

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_knstruc_HR_new.py"

BASE_OUTDIR="${SCRIPT_DIR}/results_knstruc_new"
LOGDIR="${BASE_OUTDIR}/logs"
mkdir -p "${LOGDIR}"

GPUS=(0 1 2 3)
MAX_PER_GPU=5

DATA_SEED=1
SEEDS=(1 2 3 4 5)
HIDDEN=50
EPOCHS=2500
BATCH_RATIO=0.1
USE_EARLY_STOP=false
DT_STR=001

NODE_SPECS=(
    "20   2  5  150"
    "40   2  5  100"
    "60   2  5  100"
    "80   2  2  70"
    # "100  2  2  50"
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
            SERIES_GLOB="${SCRIPT_DIR}/data_new/Series_N${NUM_NODES_KEEP}_Celegans_T500_dt${DT_STR}_gc*_d*_seed${DATA_SEED}.pickle"
            SERIES_MATCHES=(${SERIES_GLOB})
            if [ ! -e "${SERIES_MATCHES[0]}" ]; then
                echo "SKIP: missing data for N=${NUM_NODES_KEEP}, data_seed=${DATA_SEED}"
                continue
            fi
            SERIES_FILE="${SERIES_MATCHES[0]}"
            GC_TAG=$(basename "${SERIES_FILE}" | sed -E 's/^.*_gc([^_]+)_d.*$/\1/')
            DEGREE_TAG=$(basename "${SERIES_FILE}" | sed -E 's/^.*_d([^_]+)_seed.*$/\1/')
            RUN_NAME="Nodes${NUM_NODES_KEEP}_d${DEGREE_TAG}_gc${GC_TAG}_Ntrain${n_train}_hidden${HIDDEN}_seed${seed}"
            OUTDIR="${BASE_OUTDIR}/${RUN_NAME}"

            if [ -f "${OUTDIR}/results.json" ]; then
                SKIPPED=$((SKIPPED + 1))
                continue
            fi

            GPU_ID=$(wait_for_slot)
            TOTAL=$((TOTAL + 1))

            echo "[GPU ${GPU_ID}] #${TOTAL} N=${NUM_NODES_KEEP} ${RUN_NAME}"
            python "${TRAIN_SCRIPT}" \
                --num_nodes_keep "${NUM_NODES_KEEP}" \
                --num_train_samples "${n_train}" \
                --data_seed "${DATA_SEED}" \
                --seed "${seed}" \
                --hidden "${HIDDEN}" \
                --epochs "${EPOCHS}" \
                --batch_ratio "${BATCH_RATIO}" \
                --outdir "${OUTDIR}/" \
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
echo "All ${TOTAL} component-R² knstruc jobs launched (${SKIPPED} skipped). Waiting..."
wait
echo "All component-R² knstruc experiments completed."
