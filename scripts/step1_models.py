"""Шаг 1 плана: данные, Skip-GCN × 5 сидов, RF, порог τ, метрика M-10.

Запуск из корня репозитория:
    python scripts/step1_models.py
    python scripts/step1_models.py --epochs 50 --seeds 0      # быстрая проверка
"""
import os

# macOS: torch и sklearn тянут разные копии OpenMP — без этого ядро падает
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.data import describe, load_elliptic  # noqa: E402
from src.metrics import period_table, summarize_seeds  # noqa: E402
from src.model import model_path  # noqa: E402
from src.train import gcn_scores, pick_threshold, train_gcn, train_rf  # noqa: E402


def plot_f1_by_step(df: pd.DataFrame, out: Path):
    steps = df[df.period.str.startswith("шаг")].copy()
    steps["step"] = steps.period.str.extract(r"(\d+)").astype(int)
    gcn = steps[steps.model.str.startswith("gcn_seed")].groupby("step").f1.agg(["mean", "min", "max"])
    rf = steps[steps.model == "rf"].set_index("step").f1
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(gcn.index, gcn["mean"], marker="o", label="Skip-GCN (среднее по сидам)")
    ax.fill_between(gcn.index, gcn["min"], gcn["max"], alpha=0.2, label="Skip-GCN (мин–макс)")
    ax.plot(rf.index, rf.values, marker="s", ls="--", label="Random Forest")
    ax.axvline(C.SHUTDOWN_STEP - 0.5, color="grey", ls=":")
    ax.text(C.SHUTDOWN_STEP - 0.4, 0.95, "закрытие\nдаркнет-рынка", fontsize=8, va="top")
    ax.set(xlabel="шаг времени", ylabel="F1 класса illicit", ylim=(0, 1), title="M-10: качество по шагам (порог τ)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=C.DATA_ROOT)
    ap.add_argument("--out", type=Path, default=C.ARTIFACTS)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(C.SEEDS))
    ap.add_argument("--threads", type=int, default=0, help="torch.set_num_threads; 0 = по умолчанию")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    (args.out / "models").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("== Данные")
    data = load_elliptic(args.root)
    print(describe(data))

    y_val = (data.y[data.val_mask] == C.ILLICIT).numpy()
    val = data.val_mask.numpy()
    scores, taus, info = {}, {}, {}

    print(f"\n== Skip-GCN: сиды {args.seeds}, {args.epochs} эпох")
    for s in args.seeds:
        model, hist = train_gcn(data, s, epochs=args.epochs)
        torch.save(model.state_dict(), model_path(s, args.out))
        p = gcn_scores(model, data)
        tau, ok = pick_threshold(y_val, p[val])
        name = f"gcn_seed{s}"
        scores[name], taus[name] = p, tau
        info[name] = dict(tau=tau, precision_target_met=ok, history=hist)
        print(f"  {name}: τ = {tau:.4f}" + ("" if ok else "  (!) precision 0.8 на валидации недостижима, τ по max F1"))

    print("\n== Random Forest (без графа)")
    _, p_rf = train_rf(data)
    tau, ok = pick_threshold(y_val, p_rf[val])
    scores["rf"], taus["rf"] = p_rf, tau
    info["rf"] = dict(tau=tau, precision_target_met=ok)
    print(f"  rf: τ = {tau:.4f}")

    print("\n== M-10")
    df = period_table(data, scores, taus)
    df.to_csv(args.out / "m10_by_step.csv", index=False)
    summary = summarize_seeds(df)
    summary.to_csv(args.out / "m10_summary.csv", index=False)
    with pd.option_context("display.width", 140, "display.max_columns", 10):
        print(summary.to_string(index=False))
    plot_f1_by_step(df, args.out / "fig_m10_f1_by_step.png")

    np.savez_compressed(args.out / "scores.npz", **scores)
    (args.out / "thresholds.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print(f"\nГотово за {(time.time() - t0) / 60:.1f} мин. Артефакты: {args.out}")


if __name__ == "__main__":
    main()
