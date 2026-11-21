"""M-10: качество модели по шагам времени и по периодам до/после шага 43."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from . import config as C


def binary_report(y: np.ndarray, p: np.ndarray, tau: float) -> dict:
    pred = (p >= tau).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    ap = average_precision_score(y, p) if 0 < y.sum() < len(y) else np.nan
    return dict(precision=pr, recall=rc, f1=f1, pr_auc=ap, n_illicit=int(y.sum()), n_flagged=int(pred.sum()))


def period_table(data, scores: dict, taus: dict) -> pd.DataFrame:
    """scores/taus: {имя модели: p_illicit для всех узлов / порог}.
    Строки: модель × период (каждый тестовый шаг, 'test 35–49', 'до 43', 'с 43')."""
    lab = (data.y != C.UNKNOWN).numpy()
    ts = data.time_step.numpy()
    y = (data.y.numpy() == C.ILLICIT).astype(int)
    lo, hi = C.TEST_STEPS
    periods = {f"шаг {s}": ts == s for s in range(lo, hi + 1)}
    periods[f"test {lo}–{hi}"] = (ts >= lo) & (ts <= hi)
    periods[f"до {C.SHUTDOWN_STEP} ({lo}–{C.SHUTDOWN_STEP - 1})"] = (ts >= lo) & (ts < C.SHUTDOWN_STEP)
    periods[f"с {C.SHUTDOWN_STEP} ({C.SHUTDOWN_STEP}–{hi})"] = (ts >= C.SHUTDOWN_STEP) & (ts <= hi)
    rows = []
    for name, p in scores.items():
        for per, m in periods.items():
            m = m & lab
            rows.append(dict(model=name, period=per, **binary_report(y[m], p[m], taus[name])))
    return pd.DataFrame(rows)


def summarize_seeds(df: pd.DataFrame, prefix: str = "gcn_seed") -> pd.DataFrame:
    """Среднее ± std по сидам GCN для агрегированных периодов, рядом RF."""
    agg = df[~df.period.str.startswith("шаг")].copy()
    n = agg.model.str.startswith(prefix)
    agg["family"] = np.where(n, f"Skip-GCN ({agg.model[n].nunique()} сид.)", agg.model)
    cols = ["precision", "recall", "f1", "pr_auc"]
    g = agg.groupby(["family", "period"], sort=False)[cols]
    mean, std = g.mean(), g.std().fillna(0)
    out = mean.round(3).astype(str) + " ± " + std.round(3).astype(str)
    return out.reset_index()
