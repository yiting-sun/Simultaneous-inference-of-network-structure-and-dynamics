#!/bin/bash
# Known-structure dynamics inference sweep on the Rulkov discrete-map
# subgraphs. Structure is assumed known (full adjacency is handed to the
# message-passing network); only the per-node and per-edge functions are
# learned. Auto-routes per-N to data_small/ or data_large/.
#
# Re-run any time: completed configs are skipped via results.json check.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_knstruc_RulkovChemical.py"

BASE_OUTDIR="${SCRIPT_DIR}/results_knstruc"
LOGDIR="${BASE_OUTDIR}/logs"
mkdir -p "${LOGDIR}"

GPUS=(0 1 2 3)

max_per_gpu_for_N() {
  local N=$1
  if   [ "$N" -lt 100 ]; then echo 5
  elif [ "$N" -lt 160 ]; then echo 4
  else                        echo 2
  fi
}

DATA_SEED=1
SEEDS=(1 2 3 4 5)
HIDDEN=50
EPOCHS=2500
BATCH_RATIO=0.05
USE_EARLY_STOP=false
DT_STR=0001
DT_DATA=0.001
TRAIN_T_START=20
TRAIN_T_END=60
EVAL_T_START=60
EVAL_T_END=90
LR=5e-4
NUM_WORKERS=4

# Each entry: "N SEQ_START SEQ_STEP SEQ_END"
# knstruc is a small-N task only (structure is handed in, so large networks
# aren't informative here). All entries use the small branch (data_small/).
NODE_SPECS=(
    "10   5   5   102"
    "20   2   3    82"
    "30   2   3    52"
    "40   2   2    42"
    "50   2   1    35"
)

declare -A gpu_pids
for g in "${GPUS[@]}"; do gpu_pids[$g]=""; done

wait_for_slot() {
    local cap=$1
    while true; do
        for g in "${GPUS[@]}"; do
            local cnt=0
            for pid in ${gpu_pids[$g]}; do
                if kill -0 "$pid" 2>/dev/null; then cnt=$((cnt + 1)); fi
            done
            if [ "$cnt" -lt "$cap" ]; then echo "$g"; return; fi
        done
        sleep 2
    done
}

TOTAL=0
SKIPPED=0

for spec in "${NODE_SPECS[@]}"; do
    read -r N SEQ_START SEQ_STEP SEQ_END <<< "$spec"
    TRAIN_SAMPLES=($(seq ${SEQ_START} ${SEQ_STEP} ${SEQ_END}))
    MAX_PER_GPU=$(max_per_gpu_for_N "${N}")

    if [ "$N" -lt 80 ]; then
        DATA_DIR="${SCRIPT_DIR}/data_small"
    else
        DATA_DIR="${SCRIPT_DIR}/data_large"
    fi

    for n_train in "${TRAIN_SAMPLES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            SERIES_GLOB="${DATA_DIR}/Series_N${N}_RulkovChemicalODE_T100_dt${DT_STR}_gc*_d*_seed${DATA_SEED}.pickle"
            SERIES_MATCHES=(${SERIES_GLOB})
            if [ ! -e "${SERIES_MATCHES[0]}" ]; then
                echo "SKIP: missing data for N=${N} in ${DATA_DIR}"
                continue
            fi
            SERIES_FILE="${SERIES_MATCHES[0]}"
            GC_TAG=$(basename "${SERIES_FILE}" | sed -E 's/^.*_gc([^_]+)_d.*$/\1/')
            DEGREE_TAG=$(basename "${SERIES_FILE}" | sed -E 's/^.*_d([^_]+)_seed.*$/\1/')
            RUN_NAME="Nodes${N}_d${DEGREE_TAG}_gc${GC_TAG}_Ntrain${n_train}_hidden${HIDDEN}_seed${seed}"
            OUTDIR="${BASE_OUTDIR}/${RUN_NAME}"

            if [ -f "${OUTDIR}/results.json" ]; then
                SKIPPED=$((SKIPPED + 1))
                continue
            fi

            GPU_ID=$(wait_for_slot "${MAX_PER_GPU}")
            TOTAL=$((TOTAL + 1))
            echo "[GPU ${GPU_ID}] (cap=${MAX_PER_GPU}) #${TOTAL} N=${N} ${RUN_NAME}"
            python "${TRAIN_SCRIPT}" \
                --outdir "${OUTDIR}/" \
                --num_nodes_keep "${N}" \
                --num_train_samples "${n_train}" \
                --data_seed "${DATA_SEED}" \
                --seed "${seed}" \
                --hidden "${HIDDEN}" \
                --epochs "${EPOCHS}" \
                --lr "${LR}" \
                --batch_ratio "${BATCH_RATIO}" \
                --use_early_stop "${USE_EARLY_STOP}" \
                --train_t_start "${TRAIN_T_START}" \
                --train_t_end "${TRAIN_T_END}" \
                --eval_t_start "${EVAL_T_START}" \
                --eval_t_end "${EVAL_T_END}" \
                --dt_data "${DT_DATA}" \
                --num_workers "${NUM_WORKERS}" \
                --data_dir "${DATA_DIR}" \
                --device_id "${GPU_ID}" \
                > "${LOGDIR}/${RUN_NAME}.log" 2>&1 &
            gpu_pids[$GPU_ID]="${gpu_pids[$GPU_ID]} $!"
        done
    done
    echo "--- N=${N}: all jobs submitted ---"
done

echo ""
echo "knstruc sweep launched: ${TOTAL} jobs (${SKIPPED} already complete). Waiting..."
wait
echo "[$(date)] knstruc sweep DONE."
