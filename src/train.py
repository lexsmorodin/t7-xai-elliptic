"""Обучение Skip-GCN и RF, выбор порога τ."""
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve

from . import config as C
from .model import SkipGCN, p_illicit


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_gcn(data, seed: int, epochs: int = C.EPOCHS, lr: float = C.LR, log_every: int = 100):
    """Полный батч, Adam, взвешенная CE; неразмеченные узлы участвуют только в передаче сообщений."""
    set_seed(seed)
    model = SkipGCN(data.num_features)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    weight = torch.tensor(C.CLASS_WEIGHTS, dtype=torch.float)
    tr, va = data.train_mask, data.val_mask
    y_va = (data.y[va] == C.ILLICIT).numpy()
    history, t0 = [], time.time()
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[tr], data.y[tr], weight=weight)
        loss.backward()
        opt.step()
        if ep % log_every == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                p = p_illicit(model(data.x, data.edge_index))[va].numpy()
            ap = average_precision_score(y_va, p)
            history.append(dict(epoch=ep, loss=loss.item(), val_pr_auc=float(ap)))
            print(f"  seed {seed} | эпоха {ep:4d} | loss {loss:.4f} | val PR-AUC {ap:.3f} | {time.time() - t0:.0f} c",
                  flush=True)
    model.eval()
    return model, history


@torch.no_grad()
def gcn_scores(model, data) -> np.ndarray:
    return p_illicit(model(data.x, data.edge_index)).numpy()


def train_rf(data, seed: int = 0):
    """RF на всех 165 признаках, без рёбер. Обучение на тех же шагах 1–29."""
    rf = RandomForestClassifier(**C.RF_PARAMS, random_state=seed, n_jobs=-1)
    tr = data.train_mask.numpy()
    rf.fit(data.x.numpy()[tr], (data.y.numpy()[tr] == C.ILLICIT).astype(int))
    return rf, rf.predict_proba(data.x.numpy())[:, 1]


def pick_threshold(y_true: np.ndarray, p: np.ndarray, target: float = C.PRECISION_TARGET):
    """Минимальный τ (максимальный recall), при котором precision >= target.
    Если цель недостижима, берём τ с максимальным F1 и возвращаем ok=False."""
    prec, rec, thr = precision_recall_curve(y_true, p)
    prec, rec = prec[:-1], rec[:-1]
    ok = prec >= target
    if ok.any():
        i = int(np.argmax(np.where(ok, rec, -1.0)))
        return float(thr[i]), True
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[int(np.argmax(f1))]), False
