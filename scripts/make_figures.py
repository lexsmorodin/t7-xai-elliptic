"""Графики для отчёта и слайдов из артефактов шагов 1–6 → artifacts/figures/."""
import argparse
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

    # 1. M-10 по шагам: строим заново из m10_by_step.csv (тем же кодом, что шаг 1)
    import importlib.util
    spec = importlib.util.spec_from_file_location("step1", Path(__file__).with_name("step1_models.py"))
    step1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(step1)
    step1.plot_f1_by_step(pd.read_csv(a.out / "m10_by_step.csv"), fig_dir / "1_m10_f1_by_step.png")

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

    # 3. Протокол регулятора: precision групп против базы того же слоя (M-11, шаг 7)
    m11 = pd.read_csv(exp / "m11_precision_ci.csv")
    c1_layer = [s for s in m11["слой"].unique() if s.startswith("шаги C1")][0]
    panels = [
        ("35–42", "Шаги 35–42", [("все алерты слоя (база)", "все алерты"),
                                  ("C2: контрфактуал ≤ 3 потоков (вычислим без меток)", "контрфактуал\n≤ 3 потоков"),
                                  ("красный флаг: C2 и не C3 (ретроспективно)", "красный флаг\n(ретро)"),
                                  ("флаг держится без потоков", "флаг держится\nбез потоков")]),
        (c1_layer, "Шаги, где выполнено C1", [("все алерты слоя (база)", "все алерты"),
                                              ("годные по протоколу v1", "годные v1"),
                                              ("годные v1 со строгим C5", "v1, строгий\nC5"),
                                              ("v2: годные (C1–C3 у ≥ 3 из 5 сидов)", "годные v2"),
                                              ("флаг держится без потоков", "флаг держится\nбез потоков")]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw=dict(width_ratios=[4, 5]))
    for ax, (layer, title, items) in zip(axes, panels):
        sub = m11[m11["слой"] == layer].set_index("группа")
        vals, lows, highs, ns = [], [], [], []
        for g, _ in items:
            r = sub.loc[g]
            lo_, hi_ = [float(x) for x in r.ci.strip("[]").split(";")]
            vals.append(r.precision); lows.append(r.precision - lo_); highs.append(hi_ - r.precision); ns.append(r.n)
        colors = [GREY] + [BLUE if "без потоков" in n or "v2" in n else ORANGE for _, n in items[1:]]
        xs = range(len(items))
        ax.bar(xs, vals, color=colors, yerr=[lows, highs], capsize=4, ecolor="#14213D")
        for i, (v, n) in enumerate(zip(vals, ns)):
            ax.text(i, v + highs[i] + 0.03, f"{v:.2f}", ha="center", fontsize=9)
            ax.text(i, 0.03, f"n={n}", ha="center", fontsize=8, color="white")
        ax.axhline(vals[0], color=GREY, ls=":", lw=1)
        ax.set_xticks(list(xs), [n for _, n in items], fontsize=8)
        ax.set(ylim=(0, 1.1), title=title)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("precision (95% ДИ Уилсона)")
    fig.suptitle("Precision групп алертов против базы того же слоя", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "3_protocol_precision.png", dpi=160)
    plt.close(fig)

    # 4. Сигналы для аналитика (М6, из M-11 шага 7)
    m6 = m11[m11["слой"] == "35–42"].set_index("группа")
    keys = ["М6 A: без потоков ✓, прецеденты ✓", "М6 B: без потоков ✓, прецеденты ✗",
            "М6 C: без потоков ✗, прецеденты ✓", "М6 D: без потоков ✗, прецеденты ✗"]
    names = ["без потоков ✓\nпрецеденты ✓", "без потоков ✓\nпрецеденты ✗", "без потоков ✗\nпрецеденты ✓",
             "без потоков ✗\nпрецеденты ✗"]
    vals = [float(m6.loc[k, "precision"]) for k in keys]
    ns = [int(m6.loc[k, "n"]) for k in keys]
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
    bar(ax, [f"+{m}" for m in camo.index], camo.tolist(), [ORANGE] * len(camo), fmt="{:.1%}")
    ax.set(ylim=(0, 1), xlabel="потоков от случайных licit-транзакций",
           ylabel="доля снятых флагов", title="Камуфляж: верные алерты")
    fig.tight_layout()
    fig.savefig(fig_dir / "5_camouflage.png", dpi=160)
    plt.close(fig)
    print(f"Графики: {fig_dir}")


if __name__ == "__main__":
    main()
