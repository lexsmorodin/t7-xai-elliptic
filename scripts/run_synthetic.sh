#!/usr/bin/env bash
# Дымовая проверка всего конвейера на синтетике в формате Elliptic, без скачивания данных (~5 мин на CPU).
# Пишет только в data/fake_elliptic и artifacts_test/.
set -euo pipefail
cd "$(dirname "$0")/.."
export KMP_DUPLICATE_LIB_OK=TRUE
PY=${PYTHON:-python}
R="--root data/fake_elliptic --out artifacts_test"
$PY tests/make_fake_elliptic.py data/fake_elliptic
$PY scripts/step1_models.py $R --epochs 60
$PY scripts/step2_explain.py $R --limit 120
$PY scripts/step3_cf_knn.py $R --unknown-sample 20
$PY scripts/step4_faithfulness.py $R --n-stab 10 --n-plant 10
$PY scripts/step5_protocol.py $R
$PY scripts/step6_fragility.py $R --n-camo 10 --n-shop 10
$PY scripts/step7_protocol_v2.py $R
$PY scripts/make_figures.py --out artifacts_test
echo "Синтетика: все шаги прошли. Результаты в artifacts_test/"
