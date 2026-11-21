#!/usr/bin/env bash
# Полный прогон из чистого клона: bash scripts/run_all.sh  (~50 мин на CPU)
set -euo pipefail
cd "$(dirname "$0")/.."
export KMP_DUPLICATE_LIB_OK=TRUE
${PYTHON:-python} scripts/step1_models.py
${PYTHON:-python} scripts/step2_explain.py
${PYTHON:-python} scripts/step3_cf_knn.py
${PYTHON:-python} scripts/step4_faithfulness.py
${PYTHON:-python} scripts/step5_protocol.py
${PYTHON:-python} scripts/step6_fragility.py
${PYTHON:-python} scripts/step7_protocol_v2.py
${PYTHON:-python} scripts/make_figures.py
