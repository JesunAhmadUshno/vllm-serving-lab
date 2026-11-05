"""Quantized-serving comparison logic for vllm-serving-lab, Level 3 Deep.

Standard library only, so the scoring, summary, and comparison math
is unit-tested in CI with no GPU and no model download.

Scoring rule is the Build rule, kept identical on purpose: the number
after the "Final answer:" marker when present, else the last number
in the text, tolerance 0.01. Keeping the rule identical is what makes
the Deep rows comparable with the Build rows.

Configs, in report order:
- fp16: no quantization, torch_dtype float16 (the Build row).
- int8_bnb: bitsandbytes load_in_8bit.
- int4_bnb_nf4: bitsandbytes load_in_4bit, NF4.
The real loaders live in run_quant.py; this module never imports
torch or transformers.
"""
import re
import time

PROMPT_SUFFIX = (" Think step by step, then end with exactly one line: "
                 "Final answer: <number>")
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

CONFIG_ORDER = ["fp16", "int8_bnb", "int4_bnb_nf4"]
CONFIG_LABELS = {
    "fp16": "FP16 (torch_dtype float16, no quantization)",
    "int8_bnb": "INT8 (bitsandbytes load_in_8bit)",
    "int4_bnb_nf4": "INT4 (bitsandbytes load_in_4bit, NF4, compute float16)",
}
# Theoretical storage only. Never reported as measured memory.
THEORETICAL_BYTES_PER_WEIGHT = {"fp16": 2.0, "int8_bnb": 1.0,
                                "int4_bnb_nf4": 0.5}


def extract_answer(text):
    """Return the extracted float answer, or None when none is found."""
    idx = text.rfind("Final answer:")
    if idx != -1:
        tail = text[idx + len("Final answer:"):]
        match = NUMBER_RE.search(tail)
        if match:
            return float(match.group(0))
    matches = NUMBER_RE.findall(text)
    if matches:
        return float(matches[-1])
    return None


def is_correct(extracted, expected, tolerance=0.01):
    if extracted is None:
        return False
    return abs(extracted - float(expected)) <= tolerance


def theoretical_weight_bytes(param_count, config):
    """Theoretical weight storage: param count times bytes per weight."""
    return int(round(param_count * THEORETICAL_BYTES_PER_WEIGHT[config]))


def memory_saving_pct(baseline_bytes, other_bytes):
    """Percent smaller than baseline. Negative means larger."""
    if not baseline_bytes:
        return None
    return round(100.0 * (baseline_bytes - other_bytes) / baseline_bytes, 2)


def summarize(rows, wall_seconds):
    """Aggregate per-question rows for one config into a summary."""
    questions = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    completion_tokens = sum(r["completion_tokens"] for r in rows)
    return {
        "questions": questions,
        "correct": correct,
        "accuracy": round(correct / questions, 4) if questions else 0.0,
        "completion_tokens": completion_tokens,
        "wall_seconds": round(wall_seconds, 3),
        "completion_tokens_per_second": (
            round(completion_tokens / wall_seconds, 2) if wall_seconds > 0
            else None),
        "tokens_per_correct": (round(completion_tokens / correct, 2)
                               if correct else None),
        "seconds_per_correct": (round(wall_seconds / correct, 3)
                                if correct else None),
    }


def run_config(backend, questions, budget=384):
    """Run every question once at one budget through a backend.

    The backend is any object with generate(prompt, max_new_tokens)
    returning text and completion_tokens.
    """
    rows = []
    started = time.time()
    for question in questions:
        result = backend.generate(question["question"] + PROMPT_SUFFIX,
                                  budget)
        extracted = extract_answer(result["text"])
        rows.append({
            "question_id": question["id"],
            "expected": question["expected"],
            "extracted": extracted,
            "correct": is_correct(extracted, question["expected"]),
            "completion_tokens": result["completion_tokens"],
        })
    wall = time.time() - started
    return {"summary": summarize(rows, wall), "rows": rows}


def build_comparison(summaries):
    """Build the comparison table rows from per-config summaries.

    summaries maps config name to a summary dict that may carry
    gpu_bytes_after_load and peak_gpu_bytes (measured, real runs
    only). Deltas are computed against the fp16 row when present.
    """
    baseline = summaries.get("fp16") or {}
    table = []
    for config in CONFIG_ORDER:
        if config not in summaries:
            continue
        s = summaries[config]
        row = {
            "config": config,
            "label": CONFIG_LABELS[config],
            "correct": s["correct"],
            "accuracy": s["accuracy"],
            "completion_tokens": s["completion_tokens"],
            "completion_tokens_per_second":
                s["completion_tokens_per_second"],
            "tokens_per_correct": s["tokens_per_correct"],
            "gpu_bytes_after_load": s.get("gpu_bytes_after_load"),
            "peak_gpu_bytes": s.get("peak_gpu_bytes"),
            "memory_saving_pct_vs_fp16": None,
            "accuracy_delta_vs_fp16": None,
            "throughput_ratio_vs_fp16": None,
        }
        if baseline and config != "fp16":
            if s.get("gpu_bytes_after_load") and \
                    baseline.get("gpu_bytes_after_load"):
                row["memory_saving_pct_vs_fp16"] = memory_saving_pct(
                    baseline["gpu_bytes_after_load"],
                    s["gpu_bytes_after_load"])
            row["accuracy_delta_vs_fp16"] = round(
                s["accuracy"] - baseline["accuracy"], 4)
            if s["completion_tokens_per_second"] and \
                    baseline["completion_tokens_per_second"]:
                row["throughput_ratio_vs_fp16"] = round(
                    s["completion_tokens_per_second"]
                    / baseline["completion_tokens_per_second"], 3)
        table.append(row)
    return table


class FakeQuantBackend:
    """Deterministic backend for tests and CI. Not a model.

    Answers the questions it is told to answer correctly (by id) and
    answers the rest wrong by a fixed offset. Token counts are fixed
    per question so two runs produce identical summaries apart from
    wall time, which the deterministic runner does not record.
    """

    name = "fake-quant"

    def __init__(self, questions, correct_ids):
        self._by_text = {q["question"]: q for q in questions}
        self._correct_ids = set(correct_ids)

    def count_tokens(self, text):
        return len(text.split())

    def generate(self, prompt, max_new_tokens):
        question = None
        for text, q in self._by_text.items():
            if prompt.startswith(text):
                question = q
                break
        if question is None:
            raise ValueError("fake backend got an unknown prompt")
        if question["id"] in self._correct_ids:
            answer = question["expected"]
        else:
            answer = float(question["expected"]) + 7
        text = ("step by step work. Final answer: %s" % answer)
        return {"text": text,
                "prompt_tokens": self.count_tokens(prompt),
                "completion_tokens": 200}
