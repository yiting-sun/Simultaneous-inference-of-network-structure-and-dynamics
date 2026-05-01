#!/bin/bash
# Step 1 of the struc-only experiment.
#
# For each (msg_config, adj_type) pair below, train the fixed-adj
# msg-ablation model on the FULL training data (60% of all data, no
# uniform subsampling), with sub-node selection driven by --seed.
#
# All four reference npzs live under ../datalimits4dyn_nozscore/.
#
# Sub-node sweep:
#   * N in {5, 10, 15}: run 5 seeds (different subgraphs per seed)
#   * N = 20:           run 1 seed only (the full graph; subgraph identity
#                       is irrelevant once you keep all nodes)
#
# Output goes to runs_struc_msg_nodes${N}/${RUN_NAME}/.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_fixed_adj_struc_subnodes.py"

GPUS=(0 1 2 3)
MAX_PER_GPU=5

MSG_CONFIGS=(rate_x)
ADJ_TYPES=(runs3thr0p6)
NODE_SIZES=(13 15 17 20)

# Per-N seed list. N=20 only needs one seed.
seeds_for_n() {
    local n="$1"
    if [ "$n" -ge 20 ]; then
        echo "1"
    else
        echo "1 2 3 4 5"
    fi
}

BATCH_RATIO=0.1
EPOCHS=2000

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

for N in "${NODE_SIZES[@]}"; do
    BASE_OUTDIR="${SCRIPT_DIR}/runs_struc_msg_nodes${N}"
    LOGDIR="${BASE_OUTDIR}/logs"
    mkdir -p "${LOGDIR}"

    SEEDS=$(seeds_for_n "$N")

    for msg in "${MSG_CONFIGS[@]}"; do
        for adj_type in "${ADJ_TYPES[@]}"; do
        for seed in $SEEDS; do
            RUN_NAME="run${seed}_${msg}_${adj_type}_nozscore_gaussian"
            OUTDIR="${BASE_OUTDIR}/${RUN_NAME}"

            if [ -f "${OUTDIR}/best_model.pt" ] && [ -f "${OUTDIR}/results.json" ]; then
                echo "SKIP: ${RUN_NAME} (N=${N})"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi

            GPU_ID=$(wait_for_slot)
            TOTAL=$((TOTAL + 1))

            echo "[GPU ${GPU_ID}] #${TOTAL} START: ${RUN_NAME} (N=${N})"
            python "${TRAIN_SCRIPT}" \
                --outdir "${OUTDIR}" \
                --adj_type "${adj_type}" \
                --msg_config "${msg}" \
                --zscore false \
                --smooth gaussian \
                --hidden 100 \
                --epochs "${EPOCHS}" \
                --batch_ratio "${BATCH_RATIO}" \
                --lr 1e-3 \
                --early_stop_patience 40 \
                --use_seed true \
                --seed "${seed}" \
                --device "cuda:${GPU_ID}" \
                --train_ratio 0.6 \
                --val_ratio 0.2 \
                --test_ratio 0.2 \
                --num_nodes_keep "${N}" \
                > "${LOGDIR}/${RUN_NAME}.log" 2>&1 &

            gpu_pids[$GPU_ID]="${gpu_pids[$GPU_ID]} $!"
        done
        done
    done
done

echo ""
echo "All ${TOTAL} jobs launched (${SKIPPED} skipped). Waiting for completion..."
wait
echo "Step 1 done: struc fixed-adj sub-nodes sweep finished."
