"""Uncertainty for the recorded accuracy numbers. Stdlib only, CPU only.

Wilson 95% intervals for each accuracy, and an exact two-sided McNemar
test on the paired per-question rows (same 10 questions, two configs).
Reads the committed result files; runs no model.

Usage: python3 level-3-deep/significance.py
Prints the table and writes level-3-deep/results_significance.json
(rounded to 4 places, byte-identical across runs; CI checks it).
"""
import json
import math
import os

Z95 = 1.959964


def wilson(correct, n, z=Z95):
    """Wilson score interval for correct / n."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = correct / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = (z / den) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(b, c):
    """Exact two-sided McNemar p-value from the two discordant counts.

    b: questions only config A got right; c: questions only config B got right.
    Under the null each discordant pair is a fair coin, so the p-value is
    2 * P(X <= min(b, c)) for X ~ Binomial(b + c, 0.5), capped at 1.
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def discordant(rows_a, rows_b):
    """(only A correct, only B correct) over questions present in both."""
    a = {r["question_id"]: r["correct"] for r in rows_a}
    b = {r["question_id"]: r["correct"] for r in rows_b}
    shared = sorted(set(a) & set(b))
    return (sum(a[q] and not b[q] for q in shared), sum(b[q] and not a[q] for q in shared))


def load_paired(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["rows"]


def significance_report(root):
    quant = load_paired(os.path.join(root, "level-3-deep", "results_quant.json"))
    build = {}
    for r in load_paired(os.path.join(root, "level-2-build", "results_real.json")):
        build.setdefault(f"budget_{r['budget']}", []).append(r)
    groups = {**quant, **build}
    pairs = [("fp16", "int8_bnb"), ("fp16", "int4_bnb_nf4"), ("int8_bnb", "int4_bnb_nf4"),
             ("budget_128", "budget_384"), ("budget_32", "budget_384")]
    out = {"wilson": {}, "paired": []}
    for name, rows in groups.items():
        out["wilson"][name] = wilson(sum(r["correct"] for r in rows), len(rows))
    for a, b in pairs:
        only_a, only_b = discordant(groups[a], groups[b])
        out["paired"].append({"a": a, "b": b, "only_a": only_a, "only_b": only_b,
                              "p_value": mcnemar_exact(only_a, only_b)})
    return out


def rounded(rep):
    return {"wilson_95": {k: [round(lo, 4), round(hi, 4)] for k, (lo, hi) in rep["wilson"].items()},
            "paired_exact_mcnemar": [{**p, "p_value": round(p["p_value"], 4)} for p in rep["paired"]]}


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    rep = significance_report(os.path.dirname(here))
    with open(os.path.join(here, "results_significance.json"), "w", encoding="utf-8") as f:
        json.dump(rounded(rep), f, indent=2, sort_keys=True)
        f.write("\n")
    for name, (lo, hi) in rep["wilson"].items():
        print(f"{name}: Wilson 95% {lo:.3f} to {hi:.3f}")
    for p in rep["paired"]:
        print(f"{p['a']} vs {p['b']}: only A {p['only_a']}, only B {p['only_b']}, exact McNemar p = {p['p_value']:.4f}")
