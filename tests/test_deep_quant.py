"""Deep tests: quantization comparison scoring and summary math.

No GPU, no model download: the fake backend and fabricated summaries
only. Real measured rows live in level-3-deep/results_quant.json.
"""
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "level-3-deep"))

import quant_compare as qc
import run_quant


def load_questions():
    path = os.path.join(ROOT, "level-2-build", "questions.json")
    with open(path) as handle:
        return json.load(handle)


class ExtractionTests(unittest.TestCase):
    def test_prefers_final_answer(self):
        self.assertEqual(qc.extract_answer("work 99 then Final answer: 42"),
                         42.0)

    def test_fallback_last_number(self):
        self.assertEqual(qc.extract_answer("first 3 then 7"), 7.0)

    def test_none(self):
        self.assertIsNone(qc.extract_answer("no numbers here"))

    def test_tolerance(self):
        self.assertTrue(qc.is_correct(74.8, 74.80))
        self.assertFalse(qc.is_correct(75.0, 74.80))
        self.assertFalse(qc.is_correct(None, 1))


class SummaryMathTests(unittest.TestCase):
    def test_tokens_per_correct_precomputed(self):
        rows = [{"correct": True, "completion_tokens": 300},
                {"correct": True, "completion_tokens": 330},
                {"correct": False, "completion_tokens": 384}]
        s = qc.summarize(rows, 20.0)
        self.assertEqual(s["correct"], 2)
        self.assertEqual(s["completion_tokens"], 1014)
        self.assertEqual(s["tokens_per_correct"], 507.0)
        self.assertEqual(s["seconds_per_correct"], 10.0)
        self.assertEqual(s["completion_tokens_per_second"], 50.7)
        self.assertEqual(s["accuracy"], round(2 / 3, 4))

    def test_no_correct_gives_none_costs(self):
        rows = [{"correct": False, "completion_tokens": 384}]
        s = qc.summarize(rows, 5.0)
        self.assertIsNone(s["tokens_per_correct"])
        self.assertIsNone(s["seconds_per_correct"])
        self.assertEqual(s["accuracy"], 0.0)

    def test_memory_saving_pct(self):
        self.assertEqual(qc.memory_saving_pct(1000, 550), 45.0)
        self.assertEqual(qc.memory_saving_pct(1000, 1000), 0.0)
        self.assertEqual(qc.memory_saving_pct(1000, 1200), -20.0)
        self.assertIsNone(qc.memory_saving_pct(0, 10))

    def test_theoretical_weight_bytes_labelled(self):
        self.assertEqual(qc.theoretical_weight_bytes(1000, "fp16"), 2000)
        self.assertEqual(qc.theoretical_weight_bytes(1000, "int8_bnb"), 1000)
        self.assertEqual(qc.theoretical_weight_bytes(1000, "int4_bnb_nf4"),
                         500)


class ComparisonTests(unittest.TestCase):
    def _summaries(self):
        base = {"questions": 10, "wall_seconds": 50.0,
                "completion_tokens": 2500, "seconds_per_correct": 6.25}
        return {
            "fp16": dict(base, correct=8, accuracy=0.8,
                         completion_tokens_per_second=50.0,
                         tokens_per_correct=312.5,
                         gpu_bytes_after_load=1000, peak_gpu_bytes=1100),
            "int8_bnb": dict(base, correct=7, accuracy=0.7,
                             completion_tokens_per_second=40.0,
                             tokens_per_correct=357.14,
                             gpu_bytes_after_load=600, peak_gpu_bytes=700),
            "int4_bnb_nf4": dict(base, correct=6, accuracy=0.6,
                                 completion_tokens_per_second=80.0,
                                 tokens_per_correct=416.67,
                                 gpu_bytes_after_load=400,
                                 peak_gpu_bytes=500),
        }

    def test_order_and_deltas(self):
        table = qc.build_comparison(self._summaries())
        self.assertEqual([r["config"] for r in table], qc.CONFIG_ORDER)
        self.assertEqual(table[1]["memory_saving_pct_vs_fp16"], 40.0)
        self.assertEqual(table[1]["accuracy_delta_vs_fp16"], -0.1)
        self.assertEqual(table[2]["throughput_ratio_vs_fp16"], 1.6)
        self.assertIsNone(table[0]["memory_saving_pct_vs_fp16"])

    def test_missing_config_skipped(self):
        summaries = self._summaries()
        del summaries["int8_bnb"]
        table = qc.build_comparison(summaries)
        self.assertEqual([r["config"] for r in table],
                         ["fp16", "int4_bnb_nf4"])


class FakeRunTests(unittest.TestCase):
    def test_fake_run_counts_and_determinism(self):
        questions = load_questions()
        self.assertEqual(len(questions), 10)
        correct_ids = {"q01", "q02", "q07"}
        out1 = qc.run_config(qc.FakeQuantBackend(questions, correct_ids),
                             questions, 384)
        out2 = qc.run_config(qc.FakeQuantBackend(questions, correct_ids),
                             questions, 384)
        self.assertEqual(out1["rows"], out2["rows"])
        self.assertEqual(out1["summary"]["correct"], 3)
        self.assertEqual(out1["summary"]["completion_tokens"], 2000)
        self.assertEqual(out1["summary"]["tokens_per_correct"],
                         round(2000 / 3, 2))
        self.assertEqual(len(out1["rows"]), 10)

    def test_results_deep_file_is_labelled_fake(self):
        path = os.path.join(ROOT, "level-3-deep", "results_deep.json")
        if not os.path.exists(path):
            self.skipTest("run level-3-deep/run_deep.py first")
        with open(path) as handle:
            data = json.load(handle)
        self.assertEqual(data["backend"], "fake-quant")
        self.assertIn("Not a model result", data["note"])
        self.assertEqual(data["budget"], 384)


class ErrorLineTests(unittest.TestCase):
    def test_picks_error_over_progress_tail(self):
        text = ("Traceback (most recent call last):\n"
                "ValueError: boom\n"
                "Loading weights: 100%|##########| 290/290\n")
        self.assertEqual(run_quant.pick_error_line(text),
                         "ValueError: boom")

    def test_fallback_last_line_and_empty(self):
        self.assertEqual(run_quant.pick_error_line("one\ntwo\n"), "two")
        self.assertEqual(run_quant.pick_error_line(""),
                         "no output captured")


if __name__ == "__main__":
    unittest.main()
