#!/bin/bash
# Generate HR time series and true dx/dt on degree-matched Celegans subgraphs.
#
# For each (num_nodes_keep, seed), runs:
#   1. generation_HR_new.py
#   2. compute_true_dxdt_new.py
#
# All use dt=0.01, T=500, base gc=0.15.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GEN_SCRIPT="${SCRIPT_DIR}/generation_HR_new.py"
DXDT_SCRIPT="${SCRIPT_DIR}/compute_true_dxdt_new.py"

T=500
DT=0.01
GC=0.15
DATA_DIR="HR_dyn/data_new"

# Subgraph sizes to generate
NODE_SIZES=(20 40 60 80 100 120 140 160 180 200 220 240 260 279)

# Seeds (same pool as data files used by training)
SEEDS=(1 2 3)

TOTAL=0
SKIPPED=0

for n_nodes in "${NODE_SIZES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        DT_STR=$(python -c "print(format(${DT}, 'g').replace('.', ''))")
        SERIES_GLOB="${DATA_DIR}/Series_N${n_nodes}_Celegans_T${T}_dt${DT_STR}_gc*_d*_seed${seed}.pickle"
        DXDT_GLOB="${DATA_DIR}/TrueDxdt_N${n_nodes}_Celegans_T${T}_dt${DT_STR}_gc*_d*_seed${seed}.pickle"

        # Skip if both files already exist
        if compgen -G "${SERIES_GLOB}" > /dev/null && compgen -G "${DXDT_GLOB}" > /dev/null; then
            echo "SKIP: N=${n_nodes}, seed=${seed} (both files exist)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        TOTAL=$((TOTAL + 1))
        echo "===== [${TOTAL}] N=${n_nodes}, seed=${seed} ====="

        # Step 1: Generate time series
        if ! compgen -G "${SERIES_GLOB}" > /dev/null; then
            echo "  Running generation_HR_new.py ..."
            python "${GEN_SCRIPT}" \
                --num_nodes_keep "${n_nodes}" \
                --seed "${seed}" \
                --T "${T}" \
                --dt "${DT}" \
                --gc "${GC}"
        else
            echo "  Series file exists, skipping generation."
        fi

        # Step 2: Compute true dx/dt
        if ! compgen -G "${DXDT_GLOB}" > /dev/null; then
            echo "  Running compute_true_dxdt_new.py ..."
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
