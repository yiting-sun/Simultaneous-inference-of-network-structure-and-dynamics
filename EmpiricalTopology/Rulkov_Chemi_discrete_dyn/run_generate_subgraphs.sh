#!/bin/bash
# Generate Rulkov discrete-map time series and true dx/dt for sub-graphs of the
# ciona CNS network. Sub-graph strategy switches automatically inside the
# Python generator:
#    N <  80  ->  degree-matched selection,   saved to data_small/
#    N >= 80  ->  random   selection,         saved to data_large/
# The topology file must already exist (run build_ciona_cns_topology.py first)
# and be self-loop-free; the Python generator will refuse otherwise.
#
# Usage:
#    bash run_generate_subgraphs.sh              # default N list
#    bash run_generate_subgraphs.sh 20 50 100    # custom Ns
#    FORCE_REGEN=true bash run_generate_subgraphs.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GEN_SCRIPT="${SCRIPT_DIR}/generation_RulkovChemical.py"
DXDT_SCRIPT="${SCRIPT_DIR}/compute_true_dxdt_RulkovChemical.py"
DATA_SMALL="${SCRIPT_DIR}/data_small_gc0p05"
DATA_LARGE="${SCRIPT_DIR}/data_large_gc0p05"
ADJ_PATH="${SCRIPT_DIR}/data/ciona_cns_177_binary_adj.npy"

# Simulation defaults
T=100
DT=0.001
GC=0.05
SYNAPSE_TYPE="excitatory"
LAM=10.0
THETA=-1.0
DIVERGENCE_THRESHOLD=50.0
GC_BACKOFF=0.85
MAX_GC_RETRIES=20

# Which N's to simulate
if [ "$#" -gt 0 ]; then
    NODE_SIZES=("$@")
else
    NODE_SIZES=(10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 177)
fi

# Seeds (subgraph + initial state + parameter noise share a single seed)
SEEDS=(1 2)

DT_STR=$(echo "${DT}" | tr -d '.')
FORCE_REGEN="${FORCE_REGEN:-false}"

if [ ! -f "${ADJ_PATH}" ]; then
    echo "ERROR: adjacency file missing: ${ADJ_PATH}"
    echo "       Run: python ${SCRIPT_DIR}/build_ciona_cns_topology.py"
    exit 1
fi

mkdir -p "${DATA_SMALL}" "${DATA_LARGE}"

TOTAL=0
SKIPPED=0

for n_nodes in "${NODE_SIZES[@]}"; do
    # Route: small / large decides output dir
    if [ "${n_nodes}" -lt 80 ]; then
        OUT_DIR="${DATA_SMALL}"
        BRANCH="small"
    else
        OUT_DIR="${DATA_LARGE}"
        BRANCH="large"
    fi

    for seed in "${SEEDS[@]}"; do
        SERIES_GLOB="${OUT_DIR}/Series_N${n_nodes}_RulkovChemicalODE_T${T}_dt${DT_STR}_gc*_d*_seed${seed}.pickle"
        DXDT_GLOB="${OUT_DIR}/TrueDxdt_N${n_nodes}_RulkovChemicalODE_T${T}_dt${DT_STR}_gc*_d*_seed${seed}.pickle"

        if [ "${FORCE_REGEN}" = "true" ]; then
            rm -f ${SERIES_GLOB} ${DXDT_GLOB}
        fi

        if compgen -G "${SERIES_GLOB}" > /dev/null && compgen -G "${DXDT_GLOB}" > /dev/null; then
            echo "SKIP: N=${n_nodes}, seed=${seed} (${BRANCH}; series+dxdt both present)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        TOTAL=$((TOTAL + 1))
        echo "===== [${TOTAL}] N=${n_nodes}, seed=${seed}, branch=${BRANCH} ====="

        # 1. Generate series (will auto-route small/large inside Python)
        if ! compgen -G "${SERIES_GLOB}" > /dev/null; then
            echo "  [gen]  generation_RulkovChemical.py ..."
            python "${GEN_SCRIPT}" \
                --num_nodes_keep "${n_nodes}" --seed "${seed}" \
                --T "${T}" --dt "${DT}" --gc "${GC}" \
                --synapse_type "${SYNAPSE_TYPE}" --lam "${LAM}" --theta "${THETA}" \
                --divergence_threshold "${DIVERGENCE_THRESHOLD}" \
                --gc_backoff "${GC_BACKOFF}" --max_gc_retries "${MAX_GC_RETRIES}" \
                --adj_path "${ADJ_PATH}" \
                --save_dir_small "${DATA_SMALL}" --save_dir_large "${DATA_LARGE}"
        else
            echo "  [gen]  skip — series file exists."
        fi

        # 2. Compute true dx/dt
        if ! compgen -G "${DXDT_GLOB}" > /dev/null; then
            echo "  [dxdt] compute_true_dxdt_RulkovChemical.py ..."
            python "${DXDT_SCRIPT}" \
                --num_nodes_keep "${n_nodes}" --seed "${seed}" \
                --T "${T}" --dt "${DT}" \
                --synapse_type "${SYNAPSE_TYPE}" --lam "${LAM}" --theta "${THETA}" \
                --data_dir "${OUT_DIR}"
        else
            echo "  [dxdt] skip — dxdt file exists."
        fi

        echo ""
    done
done

echo "====================================="
echo "Done: ${TOTAL} generated, ${SKIPPED} skipped."
echo "      data_small/ : $(ls ${DATA_SMALL}/Series_*.pickle 2>/dev/null | wc -l) series"
echo "      data_large/ : $(ls ${DATA_LARGE}/Series_*.pickle 2>/dev/null | wc -l) series"
echo "====================================="
