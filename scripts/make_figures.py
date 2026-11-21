"""Графики для отчёта и слайдов из артефактов шагов 1–6 → artifacts/figures/."""
import argparse
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402

BLUE, ORANGE, GREY = "#2a6fdb", "#e8833a", "#9aa0a6"


def bar(ax, labels, values, colors, fmt="{:.2f}"):
    b = ax.bar(labels, values, color=colors)
    for r, v in zip(b, values):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.015, fmt.format(v), ha="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=C.ARTIFACTS)
    a = ap.parse_args()
    exp, fig_dir = a.out / "explanations" / "seed0", a.out / "figures"
    fig_dir.mkdir(exist_ok=True)
    before = f"до {C.SHUTDOWN_STEP}"

    shutil.copy(a.out / "fig_m10_f1_by_step.png", fig_dir / "1_m10_f1_by_step.png")

    # 2. Верные и ложные алерты: что держит флаг
    br = pd.read_csv(exp / "step2_breakdown.csv")
    m9 = pd.read_csv(exp / "m9_summary.csv")
    rows = []
    for lab in ["illicit", "licit"]:
        b = br[(br.period == before) & (br.label_name == lab)].iloc[0]
        m = m9[(m9.period == before) & (m9.label_name == lab)].iloc[0]
        rows.append((lab, b.holds_without_edges, m.cf_le3))
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    x = range(2)
    ax.bar([i - 0.2 for i in x], [r[1] for r in rows], 0.4, color=BLUE, label="флаг держится без потоков")
    ax.bar([i + 0.2 for i in x], [r[2] for r in rows], 0.4, color=ORANGE, label="контрфактуал ≤ 3 потоков")
    for i, r in enumerate(rows):
        ax.text(i - 0.2, r[1] + 0.02, f"{r[1]:.0%}", ha="center", fontsize=9)
        ax.text(i + 0.2, r[2] + 0.02, f"{r[2]:.0%}", ha="center", fontsize=9)
    ax.set_xticks(list(x), ["верные алерты (illicit)", "ложные алерты (licit)"])
    ax.set(ylim=(0, 1.1), ylabel="доля алертов", title=f"Шаги 35–{C.SHUTDOWN_STEP - 1}: на чём держится флаг")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "2_tp_vs_fp.png", dpi=160)
    plt.close(fig)

    # 3. Протокол регулятора: precision «годных» алертов
    pr = pd.read_csv(exp / "m5_precision.csv")
    keep = ["все размеченные алерты", "C2: контрфактуал |S| ≤ 3", "valid_doc (протокол v1)",
            "valid_strict (C5 только по меткам)", "флаг держится без рёбер"]
    names = ["все алерты", "контрфактуал\n≤ 3 потоков", "годные\n(протокол v1)", "годные\n(строгий C5)",
             "флаг держится\nбез потоков"]
    pr = pr.set_index("группа").loc[keep]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bar(ax, names, pr.precision.tolist(), [GREY, ORANGE, ORANGE, BLUE, BLUE])
    for i, n in enumerate(pr.n):
        ax.text(i, 0.03, f"n={n}", ha="center", fontsize=8, color="white")
    ax.set(ylim=(0, 1), ylabel="precision", title="Precision алертов, признанных «обоснованными»")
    fig.tight_layout()
    fig.savefig(fig_dir / "3_protocol_precision.png", dpi=160)
    plt.close(fig)

    # 4. Сигналы для аналитика
    sig = pd.read_csv(exp / "analyst_signals.csv")
    sig = sig[sig.period == before]
    order = [(True, True), (True, False), (False, True), (False, False)]
    names = ["без потоков ✓\nпрецеденты ✓", "без потоков ✓\nпрецеденты ✗", "без потоков ✗\nпрецеденты ✓",
             "без потоков ✗\nпрецеденты ✗"]
    vals, ns = [], []
    for h, k in order:
        r = sig[(sig.holds == h) & (sig.knn_major == k)]
        vals.append(float(r.precision.iloc[0]) if len(r) else 0.0)
        ns.append(int(r.n.iloc[0]) if len(r) else 0)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bar(ax, names, vals, [BLUE, BLUE, ORANGE, ORANGE])
    for i, n in enumerate(ns):
        ax.text(i, 0.03, f"n={n}", ha="center", fontsize=8, color="white")
    ax.set(ylim=(0, 1.05), ylabel="precision", title=f"Сигналы надёжности алерта (шаги 35–{C.SHUTDOWN_STEP - 1})")
    fig.tight_layout()
    fig.savefig(fig_dir / "4_analyst_signals.png", dpi=160)
    plt.close(fig)

    # 5. Камуфляж
    camo = pd.read_csv(a.out / "fragility" / "camouflage.csv").groupby("m").flag_removed.mean()
    fig, ax = plt.subplots(figsize=(5, 3.2))
    bar(ax, [f"+{m}" for m in camo.index], camo.tolist(), [ORANGE] * len(camo), fmt="{:.0%}")
    ax.set(ylim=(0, 1), xlabel="потоков от случайных licit-транзакций",
           ylabel="доля снятых флагов", title="Камуфляж: верные алерты")
    fig.tight_layout()
    fig.savefig(fig_dir / "5_camouflage.png", dpi=160)
    plt.close(fig)
    print(f"Графики: {fig_dir}")


if __name__ == "__main__":
    main()
