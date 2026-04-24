#!/bin/bash
# Generate Lorenz time series and true dx/dt on 46-node (dolphins-style) subgraphs.
#
# For each (num_nodes_keep, seed), runs:
#   1. generation_Lorenz.py      -> Series_N{N}_Lorenz_T100_dt0001_seed{s}.pickle
#   2. compute_true_dxdt_Lorenz.py -> TrueDxdt_N{N}_Lorenz_T100_dt0001_seed{s}.pickle
#
# All use dt=0.001, T=100. Coupling epsilon defaults to 0.2 for N=5 and
# 0.1 otherwise (see generation_Lorenz.py / compute_true_dxdt_Lorenz.py);
# --epsilon is not passed here so the per-N default kicks in.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GEN_SCRIPT="${SCRIPT_DIR}/generation_Lorenz.py"
DXDT_SCRIPT="${SCRIPT_DIR}/compute_true_dxdt_Lorenz.py"
DATA_DIR="${SCRIPT_DIR}/data"

T=100
DT=0.001

# Subgraph sizes to generate
NODE_SIZES=(5 10 15 20 25 30 35 40 45 50 55 58)

# Seeds (vary subgraph selection; IC seed is fixed inside the scripts)
SEEDS=(1 2 3)

TOTAL=0
SKIPPED=0

for n_nodes in "${NODE_SIZES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        DT_STR=$(echo $DT | tr -d '.')
        SERIES_FILE="${DATA_DIR}/Series_N${n_nodes}_Lorenz_T${T}_dt${DT_STR}_seed${seed}.pickle"
        DXDT_FILE="${DATA_DIR}/TrueDxdt_N${n_nodes}_Lorenz_T${T}_dt${DT_STR}_seed${seed}.pickle"

        # Skip if both files already exist
        if [ -f "${SERIES_FILE}" ] && [ -f "${DXDT_FILE}" ]; then
            echo "SKIP: N=${n_nodes}, seed=${seed} (both files exist)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        TOTAL=$((TOTAL + 1))
        echo "===== [${TOTAL}] N=${n_nodes}, seed=${seed} ====="

        # Step 1: Generate time series
        if [ ! -f "${SERIES_FILE}" ]; then
            echo "  Running generation_Lorenz.py ..."
            python "${GEN_SCRIPT}" \
                --num_nodes_keep "${n_nodes}" \
                --seed "${seed}" \
                --T "${T}" \
                --dt "${DT}"
        else
            echo "  Series file exists, skipping generation."
        fi

        # Step 2: Compute true dx/dt
        if [ ! -f "${DXDT_FILE}" ]; then
            echo "  Running compute_true_dxdt_Lorenz.py ..."
            python "${DXDT_SCRIPT}" \
                --num_nodes_keep "${n_nodes}" \
                --seed "${seed}" \
                --T "${T}" \
                --dt "${DT}"
        else
            echo "  Dxdt file exists, skipping computation."
        fi

        echo ""
    done
done

echo "====================================="
echo "Done: ${TOTAL} generated, ${SKIPPED} skipped."
echo "====================================="
