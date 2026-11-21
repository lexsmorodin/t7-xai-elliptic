"""Все фиксированные решения дизайн-документа в одном месте (раздел 0 «Паспорт»)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "elliptic"
ARTIFACTS = ROOT / "artifacts"

# Временное разбиение (раздел 2): обучение / выбор порога / тест
TRAIN_STEPS = (1, 29)
VAL_STEPS = (30, 34)
TEST_STEPS = (35, 49)
SHUTDOWN_STEP = 43  # закрытие даркнет-рынка (Weber et al., 2019)

# Модель Skip-GCN (Weber et al., 2019)
HIDDEN = 100
LR = 1e-3
EPOCHS = 1000
CLASS_WEIGHTS = (0.3, 0.7)  # (licit, illicit)
SEEDS = (0, 1, 2, 3, 4)

# Порог τ: минимальный, при котором precision illicit на валидации >= цели
PRECISION_TARGET = 0.8

# Random Forest — ориентир без графа (как в Weber et al., 2019)
RF_PARAMS = dict(n_estimators=50, max_features=50)

# Метки в PyG: 0 = licit, 1 = illicit, 2 = unknown
LICIT, ILLICIT, UNKNOWN = 0, 1, 2
