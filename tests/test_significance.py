"""Unit tests for level-3-deep/significance.py (Wilson intervals, exact McNemar).

Expected values are worked by hand from the binomial distribution, then
checked against the recorded per-question rows in the result files.
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "level-3-deep"))

from significance import (  # noqa: E402
    discordant,
    load_paired,
    mcnemar_exact,
    significance_report,
    wilson,
)


class TestWilson(unittest.TestCase):
    def test_eight_of_ten(self):
        lo, hi = wilson(8, 10)
        self.assertAlmostEqual(lo, 0.4902, places=4)
        self.assertAlmostEqual(hi, 0.9433, places=4)

    def test_zero_of_ten_starts_at_zero(self):
        lo, hi = wilson(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.2775, places=4)

    def test_empty_sample_is_rejected(self):
        with self.assertRaises(ValueError):
            wilson(0, 0)


class TestMcNemar(unittest.TestCase):
    def test_four_vs_one(self):
        # 2 * (C(5,0) + C(5,1)) / 2**5 = 12 / 32
        self.assertEqual(mcnemar_exact(4, 1), 0.375)

    def test_two_vs_zero(self):
        self.assertEqual(mcnemar_exact(2, 0), 0.5)

    def test_eight_vs_zero(self):
        self.assertEqual(mcnemar_exact(0, 8), 2 / 256)

    def test_no_discordant_pairs(self):
        self.assertEqual(mcnemar_exact(0, 0), 1.0)

    def test_p_value_is_capped_at_one(self):
        self.assertEqual(mcnemar_exact(3, 3), 1.0)


class TestRecordedRows(unittest.TestCase):
    def test_fp16_vs_int4_discordant_counts(self):
        rows = load_paired(os.path.join(ROOT, "level-3-deep", "results_quant.json"))
        self.assertEqual(discordant(rows["fp16"], rows["int4_bnb_nf4"]), (4, 1))
        self.assertEqual(discordant(rows["fp16"], rows["int8_bnb"]), (2, 0))

    def test_report_matches_hand_values(self):
        rep = significance_report(ROOT)
        quant = {(r["a"], r["b"]): r for r in rep["paired"]}
        self.assertEqual(quant[("fp16", "int4_bnb_nf4")]["p_value"], 0.375)
        self.assertEqual(quant[("fp16", "int8_bnb")]["p_value"], 0.5)
        self.assertEqual(quant[("budget_128", "budget_384")]["p_value"], 2 / 256)
        self.assertAlmostEqual(rep["wilson"]["fp16"][0], 0.4902, places=4)


if __name__ == "__main__":
    unittest.main()
