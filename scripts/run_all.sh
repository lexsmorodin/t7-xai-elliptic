#!/usr/bin/env bash
# Полный прогон из чистого клона: bash scripts/run_all.sh  (~50 мин на CPU)
set -euo pipefail
cd "$(dirname "$0")/.."
export KMP_DUPLICATE_LIB_OK=TRUE
python scripts/step1_models.py
python scripts/step2_explain.py
python scripts/step3_cf_knn.py
python scripts/step4_faithfulness.py
python scripts/step5_protocol.py
python scripts/step6_fragility.py
python scripts/make_figures.py
