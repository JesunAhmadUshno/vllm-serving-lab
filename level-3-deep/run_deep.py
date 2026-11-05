"""Deterministic local run for Level 3 Deep (fake backend).

Runs the comparison mechanics over the Build question set with the
FakeQuantBackend and writes results_deep.json. Counts only: no wall
times are recorded, so two runs produce an identical file. Nothing in
this file is a model result; the real measured run is run_quant.py
and its output is results_quant.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quant_compare as qc

HERE = os.path.dirname(os.path.abspath(__file__))
QUESTIONS = os.path.join(HERE, "..", "level-2-build", "questions.json")
BUDGET = 384

# Synthetic mechanics choices for the fake run only. They exist so the
# comparison table has different rows to aggregate; they are not
# measurements and must never be quoted as model accuracy.
FAKE_CORRECT = {
    "fp16": {"q01", "q02", "q03", "q04", "q05", "q06", "q07", "q08",
             "q09"},
    "int8_bnb": {"q01", "q02", "q03", "q04", "q05", "q06", "q07",
                 "q08"},
    "int4_bnb_nf4": {"q01", "q02", "q03", "q04", "q06", "q07", "q08"},
}


def main():
    with open(QUESTIONS) as handle:
        questions = json.load(handle)
    summaries = {}
    rows = {}
    for config in qc.CONFIG_ORDER:
        backend = qc.FakeQuantBackend(questions, FAKE_CORRECT[config])
        out = qc.run_config(backend, questions, BUDGET)
        summary = dict(out["summary"])
        summary.pop("wall_seconds", None)
        summary.pop("completion_tokens_per_second", None)
        summary.pop("seconds_per_correct", None)
        summaries[config] = summary
        rows[config] = out["rows"]
    payload = {
        "backend": "fake-quant",
        "note": "Deterministic fake run for CI mechanics only. "
                "Not a model result. Real measured rows are in "
                "results_quant.json.",
        "budget": BUDGET,
        "summaries": summaries,
        "comparison": qc.build_comparison(
            {k: dict(v, completion_tokens_per_second=None)
             for k, v in summaries.items()}),
        "rows": rows,
    }
    path = os.path.join(HERE, "results_deep.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print("fake comparison counts:",
          {k: v["correct"] for k, v in summaries.items()})
    print("wrote", path)


if __name__ == "__main__":
    main()
