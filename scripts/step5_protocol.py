"""Шаг 5 плана: протокол регулятора М5, сигналы надёжности для аналитика, карточки алертов.

Условия М5 (все обязательны, пороги из документа v1):
  C1 модель работает: precision illicit размеченных алертов этого шага >= 0,7 (ретроспективно)
  C2 необходимость:   контрфактуал М4 найден и |S| <= 3
  C3 достаточность:   только top-5 рёбер М1 сохраняют флаг
  C4 устойчивость:    средний Жаккар top-5 рёбер между 5 сидами >= 0,5
  C5 проверяемость:   каждое ребро из S в пределах 2 хопов ведёт к транзакции
                      (doc)    с меткой illicit или скором >= τ — как в документе
                      (strict) только с меткой illicit — без опоры на саму модель

    python scripts/step5_protocol.py
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.data import load_elliptic  # noqa: E402
from src.explain import GraphContext, ig_edges  # noqa: E402
from src.faithfulness import jaccard  # noqa: E402
from src.model import load_model, model_path  # noqa: E402
from src.protocol import leads_to  # noqa: E402

K_TOP = 5
PREC_MIN, CF_MAX, JAC_MIN = 0.7, 3, 0.5
LABEL = {C.LICIT: "licit", C.ILLICIT: "illicit", C.UNKNOWN: "unknown"}


def top5(attr, ids):
    return set(ids[np.argsort(-attr, kind="stable")[:K_TOP]].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=C.DATA_ROOT)
    ap.add_argument("--out", type=Path, default=C.ARTIFACTS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    exp = args.out / "explanations" / f"seed{args.seed}"
    data = load_elliptic(args.root)
    taus = json.loads((args.out / "thresholds.json").read_text())
    tau = taus[f"gcn_seed{args.seed}"]["tau"]
    p_all = np.load(args.out / "scores.npz")[f"gcn_seed{args.seed}"]
    y = data.y.numpy()
    ts = data.time_step.numpy()

    cf = pd.read_csv(exp / "cf.csv")
    fm = pd.read_csv(exp / "m1_m3_per_alert.csv")[["v", "fid_minus_kept"]]
    al = pd.read_csv(exp / "alerts.csv")[["v", "m2_local", "m2_aggregated", "m2_edges", "z", "z_base"]]
    knn = pd.read_csv(exp / "knn.csv")
    ig_all = pd.read_csv(exp / "edge_ig.csv.gz")
    ig_all = ig_all[ig_all.v.isin(cf.v)]
    ig_by_v = {v: g for v, g in ig_all.groupby("v")}
    empty = ig_all.iloc[:0]
    df = cf.merge(fm, on="v").merge(al, on="v")

    # C1: precision размеченных алертов по шагу
    flag = p_all >= tau
    lab = y != C.UNKNOWN
    prec_step = {s: ((y == C.ILLICIT) & flag & lab & (ts == s)).sum() / max((flag & lab & (ts == s)).sum(), 1)
                 for s in range(C.TEST_STEPS[0], C.TEST_STEPS[1] + 1)}
    df["step_precision"] = df.step.map(prec_step)
    df["C1"] = df.step_precision >= PREC_MIN
    df["C2"] = df.cf_success & (df.cf_size <= CF_MAX)
    df["C3"] = df.fid_minus_kept.astype(bool)

    # C4: устойчивость по сидам (кеш, т.к. дорого)
    ctx = GraphContext(data)
    cache = exp / "stability_top5.csv"
    if cache.exists():
        stab = pd.read_csv(cache)
    else:
        others = [s for s in C.SEEDS if s != args.seed and model_path(s, args.out).exists()]
        print(f"== C4: IG для сидов {others} на {len(df)} алертах")
        models = {}
        for s in others:
            m = load_model(s, data.num_features, args.out)
            for p in m.parameters():
                p.requires_grad_(False)
            models[s] = m
        rows, t1 = [], time.time()
        for i, v in enumerate(df.v):
            nb = ctx.neighborhood(int(v))
            ids = nb.var_edge_ids.numpy()
            g = ig_by_v.get(v, empty).set_index("edge_id").ig
            sets = [top5(g.reindex(ids).fillna(0).to_numpy(), ids)]
            sets += [top5(ig_edges(m, data, nb)["attr"], ids) for m in models.values()]
            pairs = [jaccard(a, b) for i_, a in enumerate(sets) for b in sets[i_ + 1:]]
            rows.append(dict(v=v, jaccard_seeds=float(np.mean(pairs))))
            if (i + 1) % 200 == 0 or i + 1 == len(df):
                print(f"  {i + 1}/{len(df)} | {time.time() - t1:.0f} c", flush=True)
        stab = pd.DataFrame(rows)
        stab.to_csv(cache, index=False)
    df = df.merge(stab, on="v")
    df["C4"] = df.jaccard_seeds >= JAC_MIN

    # C5: проверяемость рёбер контрфактуала
    und = ctx.ei_und
    ei = data.edge_index.numpy()

    ok_strict = y == C.ILLICIT
    ok_doc = ok_strict | flag
    c5d, c5s = [], []
    for r in df.itertuples():
        if not r.C2:
            c5d.append(False)
            c5s.append(False)
            continue
        S = [int(e) for e in str(r.cf_removed).split()]
        c5d.append(leads_to(r.v, S, ok_doc, ei, und, data.num_nodes))
        c5s.append(leads_to(r.v, S, ok_strict, ei, und, data.num_nodes))
    df["C5_doc"], df["C5_strict"] = c5d, c5s
    base = df.C1 & df.C2 & df.C3 & df.C4
    df["valid_doc"] = base & df.C5_doc
    df["valid_strict"] = base & df.C5_strict
    df["valid_no_C1"] = df.C2 & df.C3 & df.C4 & df.C5_strict
    df.to_csv(exp / "m5_protocol.csv", index=False)

    pd.set_option("display.width", 170)
    pd.set_option("display.max_columns", 20)
    print("\n== М5: доля алертов, выполняющих условие")
    cols = ["C1", "C2", "C3", "C4", "C5_doc", "C5_strict", "valid_doc", "valid_strict", "valid_no_C1"]
    tab = df.groupby(["period", "label_name"])[cols].mean().round(3)
    tab.insert(0, "n", df.groupby(["period", "label_name"]).size())
    tab = tab.reset_index()
    tab.to_csv(exp / "m5_summary.csv", index=False)
    print(tab.to_string(index=False))

    # Главный вопрос: какова precision среди алертов, признанных «годными»?
    labd = df[df.label != C.UNKNOWN]
    rows = []
    for name, m in [("все размеченные алерты", np.ones(len(labd), bool)),
                    ("C2: контрфактуал |S| ≤ 3", labd.C2), ("valid_doc (протокол v1)", labd.valid_doc),
                    ("valid_strict (C5 только по меткам)", labd.valid_strict),
                    ("valid без C1, C5 strict", labd.valid_no_C1),
                    ("флаг держится без рёбер", labd.z_no_edges >= np.log(tau / (1 - tau)))]:
        sub = labd[np.asarray(m)]
        rows.append(dict(группа=name, n=len(sub),
                         precision=round(float((sub.label == C.ILLICIT).mean()), 3) if len(sub) else np.nan))
    pr = pd.DataFrame(rows)
    pr.to_csv(exp / "m5_precision.csv", index=False)
    print("\n== Precision среди размеченных алертов по группам")
    print(pr.to_string(index=False))

    # Сигналы надёжности для аналитика: без рёбер × чистота kNN
    pur = knn.groupby("v").nbr_label.apply(lambda s: (s == "illicit").mean()).rename("knn_purity")
    labd = labd.merge(pur, on="v")
    labd["holds"] = labd.z_no_edges >= np.log(tau / (1 - tau))
    labd["knn_major"] = labd.knn_purity >= 0.6
    sig = labd.groupby(["period", "holds", "knn_major"]).apply(
        lambda g: pd.Series(dict(n=len(g), precision=round(float((g.label == C.ILLICIT).mean()), 3))),
        include_groups=False).reset_index()
    sig.to_csv(exp / "analyst_signals.csv", index=False)
    print("\n== Сигналы для аналитика: флаг без рёбер (holds) × большинство прецедентов illicit (knn_major)")
    print(sig.to_string(index=False))

    # Карточки алертов
    cards = args.out / "cards"
    cards.mkdir(exist_ok=True)
    picks = {
        "tp_features": df[(df.label == C.ILLICIT) & (df.period.str.startswith("до")) & ~df.C2],
        "fp_one_edge": df[(df.label == C.LICIT) & (df.period.str.startswith("до")) & (df.cf_size == 1)],
        "tp_structural": df[(df.label == C.ILLICIT) & df.C2],
        "after_shutdown": df[df.period.str.startswith("с") & (df.label == C.LICIT)],
    }
    for name, sub in picks.items():
        if len(sub) == 0:
            continue
        r = sub.sort_values("p", ascending=False).iloc[0]
        v = int(r.v)
        g = ig_by_v.get(v, empty).copy()
        g = g.sort_values("ig", ascending=False).head(K_TOP)  # тот же порядок, что проверяли M-1…M-3
        lines = [f"# Алерт: транзакция {int(r.tx_id)} (шаг {int(r.step)})", "",
                 f"- Скор модели p_illicit = {r.p:.3f} (порог τ = {tau:.3f}); "
                 f"истинная метка: **{r.label_name}** (аналитику не видна)",
                 f"- Разложение М2 (сдвиг z от «типичной легальной без связей»: {r.z - r.z_base:+.2f}): "
                 f"сама транзакция {r.m2_local:+.2f}, окружение в признаках {r.m2_aggregated:+.2f}, "
                 f"окружение в графе {r.m2_edges:+.2f}",
                 f"- Без потоков окрестности флаг {'**сохраняется**' if r.z_no_edges >= np.log(tau / (1 - tau)) else 'снимается'}",
                 "", "**Потоки с наибольшим вкладом (М1)**", "",
                 "| поток | направление | вклад IG | скор соседа |", "|---|---|---|---|"]
        adj = set(ei[1, ei[0] == v].tolist()) | set(ei[0, ei[1] == v].tolist())
        for e in g.itertuples():
            if v in (e.src, e.dst):
                nbh = e.dst if e.src == v else e.src
            else:  # ребро 2-го хопа: показываем дальний конец
                nbh = e.dst if e.src in adj else e.src
            direction = "вход в алерт" if e.dst == v else ("выход из алерта" if e.src == v else "2-й хоп")
            lines.append(f"| {int(data.tx_id[e.src])} → {int(data.tx_id[e.dst])} | {direction} | {e.ig:+.2f} | "
                         f"{p_all[nbh]:.2f} |")
        kn = knn[knn.v == v]
        lines += ["", "**Похожие закрытые случаи (М3, шаги 1–29)**", "",
                  "| txId | метка | шаг | сходство |", "|---|---|---|---|"]
        lines += [f"| {k.nbr_tx_id} | {k.nbr_label} | {k.nbr_step} | {k.sim:.2f} |" for k in kn.itertuples()]
        n_ill = int((kn.nbr_label == "illicit").sum())
        holds = r.z_no_edges >= np.log(tau / (1 - tau))
        lines += ["", f"**Сигналы надёжности (М6):** флаг без потоков — {'да' if holds else 'нет'}; "
                      f"прецеденты — {n_ill} из 5 illicit → группа "
                      f"{'A' if holds and n_ill >= 3 else 'B' if holds else 'C' if n_ill >= 3 else 'D'} "
                      f"(A — проверять первым, D — последним)"]
        cf_txt = (f"убрать {int(r.cf_size)} поток(а) — флаг снимается" if r.cf_success
                  else "флаг не снимается удалением до 10 потоков: решение держится на признаках транзакции")
        lines += ["", f"**Контрфактуал (М4):** {cf_txt}", "",
                  f"**Протокол М5 v1 (ретроспективно: C1 и C5 используют метки того же шага):** "
                  f"C1 {r.C1}, C2 {r.C2}, C3 {r.C3}, C4 {r.C4}, C5 {r.C5_strict} → "
                  f"{'годное обоснование' if r.valid_strict else 'только повод для ручной проверки'}"]
        (cards / f"{name}.md").write_text("\n".join(lines))
    print(f"\nКарточки: {cards}. Готово за {(time.time() - t0) / 60:.1f} мин.")


if __name__ == "__main__":
    main()
