"""Deterministic local Build run with the FakeBackend.

Runs the thinking-budget study mechanics, an in-process server smoke
test, and a load test (counts only, no wall times) and writes
results_build.json. Two runs produce an identical file. The fake
backend is not a model: this file proves mechanics, never accuracy.
Real model numbers live in results_real.json from run_real.py.
"""
import json
import os
import sys
import threading
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import load_test
import server as server_mod
import study

HERE = os.path.dirname(os.path.abspath(__file__))
BUDGETS = [32, 128, 384]


def main():
    backend = server_mod.FakeBackend()
    questions = study.load_questions()
    study_out = study.run_study(backend, questions, BUDGETS)
    budgets_clean = {}
    for key, value in study_out["budgets"].items():
        budgets_clean[key] = {k: v for k, v in value.items()
                              if not k.startswith("_")}

    httpd = server_mod.serve(backend, port=0)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = "http://127.0.0.1:%d" % port
    try:
        with urllib.request.urlopen(base + "/health", timeout=10) as r:
            health = json.loads(r.read())
        summary, _ = load_test.run_load_test(
            base, ["What is 2 + 2?" + study.PROMPT_SUFFIX] * 8,
            concurrency=4, max_tokens=64)
    finally:
        httpd.shutdown()

    out = {
        "backend": "fake",
        "model_id": backend.model_id,
        "budgets": budgets_clean,
        "questions": len(questions),
        "health": health,
        "load_test_counts": {
            "requests": summary["requests"],
            "succeeded": summary["succeeded"],
            "failed": summary["failed"],
        },
    }
    path = os.path.join(HERE, "results_build.json")
    with open(path, "w") as handle:
        json.dump(out, handle, indent=2, sort_keys=True)
    print(json.dumps(out, indent=2, sort_keys=True))
    print("wrote", path)


if __name__ == "__main__":
    main()
