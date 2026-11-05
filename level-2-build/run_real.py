"""Serving-node run: real model, live HTTP API, measured numbers.

Starts the OpenAI-compatible server with the TransformersBackend on
the GPU, runs the thinking-budget study and the load test against
the live HTTP API (not in-process), and writes results_real.json.

Usage on the serving node:
    python run_real.py [model_id]
Default model: Qwen/Qwen2.5-0.5B-Instruct (public, no token needed).
"""
import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import load_test
import server as server_mod
import study

HERE = os.path.dirname(os.path.abspath(__file__))
BUDGETS = [32, 128, 384]
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class HttpBackend:
    """Adapter so the study talks to the live server over HTTP."""

    name = "http"
    model_id = "served-model"

    def __init__(self, base_url):
        self.base_url = base_url

    def generate(self, prompt, max_new_tokens):
        import urllib.request
        body = json.dumps({
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
        }).encode()
        req = urllib.request.Request(
            self.base_url + "/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as response:
            payload = json.loads(response.read())
        return {
            "text": payload["choices"][0]["message"]["content"],
            "prompt_tokens": payload["usage"]["prompt_tokens"],
            "completion_tokens": payload["usage"]["completion_tokens"],
        }


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    model_id = args[0] if args else DEFAULT_MODEL
    stop = "--stop" in sys.argv
    backend = server_mod.TransformersBackend(model_id, stop_final_answer=stop)
    httpd = server_mod.serve(backend, host="127.0.0.1", port=8765)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = "http://127.0.0.1:8765"
    try:
        questions = study.load_questions()
        study_out = study.run_study(HttpBackend(base), questions, BUDGETS)
        prompts = [q["question"] + study.PROMPT_SUFFIX for q in questions[:6]]
        summary, _ = load_test.run_load_test(base, prompts * 2,
                                             concurrency=4, max_tokens=128)
    finally:
        httpd.shutdown()
    out = {
        "backend": "transformers",
        "model_id": model_id,
        "device": backend.device_name,
        "stop_final_answer": stop,
        "budgets": study_out["budgets"],
        "rows": study_out["rows"],
        "load_test": summary,
    }
    path = os.path.join(HERE, "results_real_stop.json" if stop
                        else "results_real.json")
    with open(path, "w") as handle:
        json.dump(out, handle, indent=2)
    print(json.dumps({"budgets": study_out["budgets"],
                      "load_test": summary}, indent=2), flush=True)
    print("wrote", path, flush=True)
    httpd.server_close()
    os._exit(0)  # daemon server threads must not block interpreter shutdown


if __name__ == "__main__":
    main()
