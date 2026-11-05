"""Serving-node run for Level 3 Deep: the quantized comparison.

Each precision config runs in its own subprocess, so a failed load
cannot take the other rows down and GPU memory starts clean for
every row. The parent aggregates the per-config partial files into
results_quant.json and records every failed route with the exact
error line from its subprocess.

Usage on the serving node (BlackBox):
    python run_quant.py                 # all configs, aggregate
    python run_quant.py --worker fp16   # one config, partial file

Configs: fp16 (no quantization), int8_bnb (bitsandbytes
load_in_8bit), int4_bnb_nf4 (bitsandbytes load_in_4bit, NF4).
Model: Qwen/Qwen2.5-0.5B-Instruct (public, no token used).
Budget: 384 new tokens, greedy, the Build question set and scoring.
Memory: torch.cuda memory_allocated after load, and
max_memory_allocated across the run. Same counters for every row.
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quant_compare as qc

HERE = os.path.dirname(os.path.abspath(__file__))
QUESTIONS = os.path.join(HERE, "..", "level-2-build", "questions.json")
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
BUDGET = 384


class TransformersQuantBackend:
    """One real model loaded at one precision. Node only."""

    def __init__(self, config):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        started = time.time()
        if config == "fp16":
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, torch_dtype=torch.float16).to("cuda")
        elif config == "int8_bnb":
            from transformers import BitsAndBytesConfig
            cfg = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, quantization_config=cfg, device_map="cuda")
        elif config == "int4_bnb_nf4":
            from transformers import BitsAndBytesConfig
            cfg = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16)
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, quantization_config=cfg, device_map="cuda")
        else:
            raise ValueError("unknown config " + config)
        self.load_seconds = round(time.time() - started, 3)
        self.device_name = torch.cuda.get_device_name(0)
        self.gpu_bytes_after_load = int(torch.cuda.memory_allocated())
        torch.cuda.reset_peak_memory_stats()

    @property
    def peak_gpu_bytes(self):
        return int(self._torch.cuda.max_memory_allocated())

    def generate(self, prompt, max_new_tokens):
        torch = self._torch
        messages = [{"role": "user", "content": prompt}]
        text_in = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text_in, return_tensors="pt").to("cuda")
        prompt_tokens = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id)
        gen = out[0][prompt_tokens:]
        text = self.tokenizer.decode(gen, skip_special_tokens=True)
        return {"text": text, "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(len(gen))}


def worker(config):
    import torch
    import transformers
    try:
        import bitsandbytes
        bnb_version = bitsandbytes.__version__
    except Exception:
        bnb_version = None
    backend = TransformersQuantBackend(config)
    with open(QUESTIONS) as handle:
        questions = json.load(handle)
    out = qc.run_config(backend, questions, BUDGET)
    summary = dict(out["summary"])
    summary["gpu_bytes_after_load"] = backend.gpu_bytes_after_load
    summary["peak_gpu_bytes"] = backend.peak_gpu_bytes
    summary["load_seconds"] = backend.load_seconds
    payload = {
        "config": config,
        "label": qc.CONFIG_LABELS[config],
        "model_id": MODEL_ID,
        "device": backend.device_name,
        "versions": {"torch": torch.__version__,
                     "transformers": transformers.__version__,
                     "bitsandbytes": bnb_version,
                     "python": sys.version.split()[0]},
        "budget": BUDGET,
        "summary": summary,
        "rows": out["rows"],
    }
    path = os.path.join(HERE, "partial_%s.json" % config)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print("WORKER DONE", config, summary, flush=True)
    # Torch and transformers threads can fail interpreter shutdown
    # after the partial file is safely written (observed on this
    # node for the fp16 worker: "Fatal Python error:
    # _enter_buffered_busy" at finalizing, partial already on disk).
    # Exit hard, the Build runner does the same for its server
    # threads.
    os._exit(0)


def pick_error_line(text):
    """Best error line from a failed worker's output.

    Prefers the last line naming an error, because progress bars
    and warnings can be the literal last line of the stream.
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return "no output captured"
    for line in reversed(lines):
        if "error" in line.lower() or "traceback" in line.lower():
            return line
    return lines[-1]


def parent():
    summaries = {}
    per_config = {}
    failed = []
    for config in qc.CONFIG_ORDER:
        partial = os.path.join(HERE, "partial_%s.json" % config)
        if os.path.exists(partial):
            os.remove(partial)
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--worker", config],
            capture_output=True, text=True, timeout=1800)
        if os.path.exists(partial):
            with open(partial) as handle:
                data = json.load(handle)
            if proc.returncode != 0:
                data["worker_returncode"] = proc.returncode
                data["worker_note"] = (
                    "worker exited %d after writing its partial file; "
                    "the partial data was complete and is used"
                    % proc.returncode)
            per_config[config] = data
            summaries[config] = data["summary"]
            os.remove(partial)
            print("OK", config, data["summary"], flush=True)
        else:
            error_line = pick_error_line(proc.stderr or proc.stdout)
            failed.append({"config": config,
                           "label": qc.CONFIG_LABELS[config],
                           "returncode": proc.returncode,
                           "error": error_line})
            print("FAILED", config, error_line, flush=True)
    first = next(iter(per_config.values()), {})
    payload = {
        "backend": "transformers+bitsandbytes",
        "model_id": MODEL_ID,
        "device": first.get("device"),
        "versions": first.get("versions"),
        "budget": BUDGET,
        "questions_file": "level-2-build/questions.json",
        "scoring": "Build rule: Final answer marker, else last number, "
                   "tolerance 0.01",
        "memory_method": "torch.cuda memory_allocated after load; "
                         "max_memory_allocated across the 10 questions",
        "summaries": summaries,
        "comparison": qc.build_comparison(summaries),
        "rows": {k: v["rows"] for k, v in per_config.items()},
        "load_seconds": {k: v["summary"].get("load_seconds")
                         for k, v in per_config.items()},
        "worker_notes": {k: v["worker_note"] for k, v in
                         per_config.items() if "worker_note" in v},
        "failed_routes": failed,
    }
    path = os.path.join(HERE, "results_quant.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        parent()
