"""Thinking-budget study for vllm-serving-lab, Level 2 Build.

The same hard question set is run at three generation budgets.
Scoring is by verifiable numeric answer: the number after the
"Final answer:" marker when present, else the last number in the
text, compared with tolerance 0.01. Cost per correct answer is
reported in completion tokens and in wall seconds.
"""
import json
import os
import re
import time

PROMPT_SUFFIX = (" Think step by step, then end with exactly one line: "
                 "Final answer: <number>")
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def load_questions(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "questions.json")
    with open(path) as handle:
        return json.load(handle)


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


def run_study(backend, questions, budgets):
    """Run every question at every budget through a backend.

    The backend is any object with generate(prompt, max_new_tokens)
    returning text, prompt_tokens, and completion_tokens (the server
    backends, or the HTTP client adapter in run_real.py).
    """
    per_budget = {}
    rows = []
    for budget in budgets:
        correct = 0
        completion_tokens = 0
        started = time.time()
        for question in questions:
            result = backend.generate(question["question"] + PROMPT_SUFFIX,
                                      budget)
            extracted = extract_answer(result["text"])
            ok = is_correct(extracted, question["expected"])
            correct += 1 if ok else 0
            completion_tokens += result["completion_tokens"]
            rows.append({
                "budget": budget,
                "question_id": question["id"],
                "expected": question["expected"],
                "extracted": extracted,
                "correct": ok,
                "completion_tokens": result["completion_tokens"],
            })
        seconds = time.time() - started
        per_budget[str(budget)] = {
            "questions": len(questions),
            "correct": correct,
            "accuracy": round(correct / len(questions), 4) if questions else 0.0,
            "completion_tokens": completion_tokens,
            "tokens_per_correct": (round(completion_tokens / correct, 2)
                                   if correct else None),
            "seconds_per_correct": (round(seconds / correct, 3)
                                    if correct else None),
            "_seconds_total": round(seconds, 3),
        }
    return {"budgets": per_budget, "rows": rows}
