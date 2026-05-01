#!/bin/bash
# Step 2 of the struc-only experiment.
#
# For each (seed, num_nodes_keep, msg_config, num_train_samples) combination,
# train the structure-only model: Step-1's self/msg MLPs are loaded and
# frozen, only edge_logits are learned (softmax + tau decay + L1 sparsity).
#
# Sub-node sweep:
#   * N in {5, 10, 15}: 5 seeds (different sub-graphs per seed)
#   * N = 20:           1 seed only (full graph)
#
# Output goes to runs_struc_only_msg_nodes${N}/${RUN_NAME}/.
# RUN_NAME embeds adj_type, lam and tau_decay so different sweeps don't collide.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_struc_only_subnodes.py"

GPUS=(0 1 2 3)
MAX_PER_GPU=12

MSG_CONFIGS=(rate_x)
ADJ_TYPES=(runs3thr0p6)
NODE_SIZES=(13 15 17 20)

# Per-N sub-graph seed list. N=20 only needs one sub-graph seed (matches Step 1).
seeds_for_n() {
    local n="$1"
    if [ "$n" -ge 20 ]; then
        echo "1"
    else
        echo "1 2 3 4 5"
    fi
}

# NN-initialization seeds run for EVERY (sub-graph seed, ...) combination.
INIT_SEEDS=(1 2 3 4 5)

# Hyperparams that go into the run name (so sweeps don't collide).
LAMS=(0.001)
TAU_DECAYS=(0.99)
TAU=1.0

# Per-node training-data sizes to sweep (uniform subsample, like simul).
# Different N can use different sweeps — edit the cases below as needed.
num_train_samples_for_n() {
    local n="$1"
    case "$n" in
        13) seq 1 1 30 ;;
        15) seq 2 2 30 ;;
        17) seq 1 1 40 ;;
        20) seq 2 2 40 ;;
        *)  seq 1 2 40 ;;
    esac
}

BATCH_RATIO=0.1
EPOCHS=2000
HIDDEN=100
DROPOUT=0.10

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
MISSING_PRETRAINED=0

for N in "${NODE_SIZES[@]}"; do
    BASE_OUTDIR="${SCRIPT_DIR}/runs_struc_only_msg_nodes${N}"
    LOGDIR="${BASE_OUTDIR}/logs"
    mkdir -p "${LOGDIR}"

    SEEDS=$(seeds_for_n "$N")
    N_TRAIN_LIST=$(num_train_samples_for_n "$N")

    for msg in "${MSG_CONFIGS[@]}"; do
        for adj_type in "${ADJ_TYPES[@]}"; do
        for lam in "${LAMS[@]}"; do
            for tau_decay in "${TAU_DECAYS[@]}"; do
                for n_train in $N_TRAIN_LIST; do
                    for seed in $SEEDS; do
                        # Resolve the matching Step-1 checkpoint. For N=20 the
                        # Step-1 sweep only ran seed=1, so we hard-code it here.
                        if [ "$N" -ge 20 ]; then
                            STEP1_SEED=1
                        else
                            STEP1_SEED="$seed"
                        fi
                        STEP1_RUN="run${STEP1_SEED}_${msg}_${adj_type}_nozscore_gaussian"
                        STEP1_PT="${SCRIPT_DIR}/runs_struc_msg_nodes${N}/${STEP1_RUN}/best_model.pt"
                        if [ ! -f "${STEP1_PT}" ]; then
                            echo "MISSING Step-1 checkpoint, skipping: ${STEP1_PT}"
                            MISSING_PRETRAINED=$((MISSING_PRETRAINED + 1))
                            continue
                        fi

                        for init_seed in "${INIT_SEEDS[@]}"; do
                            RUN_NAME="run${seed}_s${init_seed}_${msg}_${adj_type}_lam${lam}_taudecay${tau_decay}_nozscore_gaussian_n${n_train}"
                            OUTDIR="${BASE_OUTDIR}/${RUN_NAME}"

                            if [ -f "${OUTDIR}/results.json" ] && [ -f "${OUTDIR}/best_inferred_adjacency.npy" ]; then
                                echo "SKIP: ${RUN_NAME} (N=${N})"
                                SKIPPED=$((SKIPPED + 1))
                                continue
                            fi

                            GPU_ID=$(wait_for_slot)
                            TOTAL=$((TOTAL + 1))

                            echo "[GPU ${GPU_ID}] #${TOTAL} START: ${RUN_NAME} (N=${N})"
                            python "${TRAIN_SCRIPT}" \
                                --outdir "${OUTDIR}" \
                                --msg_config "${msg}" \
                                --adj_type "${adj_type}" \
                                --pretrained_pt "${STEP1_PT}" \
                                --zscore false \
                                --smooth gaussian \
                                --hidden "${HIDDEN}" \
                                --dropout "${DROPOUT}" \
                                --epochs "${EPOCHS}" \
                                --batch_ratio "${BATCH_RATIO}" \
                                --lr 1e-3 \
                                --early_stop_patience 40 \
                                --use_seed true \
                                --seed "${seed}" \
                                --init_seed "${init_seed}" \
                                --device "cuda:${GPU_ID}" \
                                --train_ratio 0.6 \
                                --val_ratio 0.2 \
                                --test_ratio 0.2 \
                                --tau "${TAU}" \
                                --tau_decay "${tau_decay}" \
                                --lam "${lam}" \
                                --num_train_samples "${n_train}" \
                                --num_nodes_keep "${N}" \
                                > "${LOGDIR}/${RUN_NAME}.log" 2>&1 &

                            gpu_pids[$GPU_ID]="${gpu_pids[$GPU_ID]} $!"
                        done
                    done
                done
            done
        done
        done
    done
done

echo ""
echo "All ${TOTAL} jobs launched (${SKIPPED} skipped, ${MISSING_PRETRAINED} missing Step-1 checkpoints)."
wait
echo "Step 2 done: struc-only sub-nodes sweep finished."
