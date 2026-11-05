"""Hand-computed unit tests for P03 Learn serving_sim. Stdlib unittest."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "level-1-learn"))

from serving_sim import (  # noqa: E402
    EXAMPLE_CONFIG,
    ModelConfig,
    RequestRecord,
    contiguous_allocation,
    kv_bytes_for_request,
    kv_bytes_for_tokens,
    kv_bytes_per_token,
    paged_allocation,
    simulate_batch,
)


class TestKVCache(unittest.TestCase):
    def test_kv_bytes_per_token_hand_computed(self):
        # 2 * 32 * 8 * 128 * 2 = 131072
        self.assertEqual(kv_bytes_per_token(EXAMPLE_CONFIG), 131072)

    def test_kv_bytes_per_token_small_config(self):
        cfg = ModelConfig(name="tiny", layers=2, kv_heads=1, head_dim=4,
                           bytes_per_element=2)
        # 2 * 2 * 1 * 4 * 2 = 32
        self.assertEqual(kv_bytes_per_token(cfg), 32)

    def test_kv_bytes_for_tokens(self):
        # 100 * 131072 = 13107200
        self.assertEqual(kv_bytes_for_tokens(100), 13107200)

    def test_kv_bytes_for_request(self):
        r = RequestRecord("X", prompt_tokens=120, thinking_tokens=200,
                          answer_tokens=80)
        # total 400, 400 * 131072 = 52428800
        self.assertEqual(r.total_tokens, 400)
        self.assertEqual(kv_bytes_for_request(r), 52428800)


class TestTokenSplit(unittest.TestCase):
    def test_thinking_answer_split_reasoning(self):
        r = RequestRecord("R", 10, 7, 3)
        self.assertEqual(r.output_tokens, 10)
        self.assertEqual(r.total_tokens, 20)
        self.assertTrue(r.is_reasoning)

    def test_thinking_zero_non_reasoning(self):
        r = RequestRecord("N", 60, 0, 45)
        self.assertEqual(r.output_tokens, 45)
        self.assertFalse(r.is_reasoning)

    def test_log_line_separates_thinking_from_answer(self):
        r = RequestRecord("R01", 120, 200, 80)
        line = r.log_line()
        self.assertIn("thinking=200", line)
        self.assertIn("answer=80", line)
        self.assertIn("prompt=120", line)
        self.assertLess(line.index("thinking=200"), line.index("answer=80"))

    def test_split_sums_across_requests(self):
        records = [RequestRecord("A", 10, 5, 5),
                   RequestRecord("B", 20, 0, 10),
                   RequestRecord("C", 5, 15, 0)]
        self.assertEqual(sum(r.thinking_tokens for r in records), 20)
        self.assertEqual(sum(r.answer_tokens for r in records), 15)
        self.assertEqual(sum(r.total_tokens for r in records), 70)


class TestBlockAllocation(unittest.TestCase):
    def test_paged_exact_block_multiple(self):
        # 400 / 16 = 25 blocks exactly, waste 0
        a = paged_allocation(400, block_size=16)
        self.assertEqual(a["blocks"], 25)
        self.assertEqual(a["allocated_tokens"], 400)
        self.assertEqual(a["waste_tokens"], 0)

    def test_paged_partial_block_waste(self):
        # 105 tokens: ceil(105/16)=7 blocks, 112 allocated, waste 7
        a = paged_allocation(105, block_size=16)
        self.assertEqual(a["blocks"], 7)
        self.assertEqual(a["allocated_tokens"], 112)
        self.assertEqual(a["waste_tokens"], 7)

    def test_paged_small_request(self):
        # 55 tokens: ceil(55/16)=4 blocks, 64 allocated, waste 9
        a = paged_allocation(55, block_size=16)
        self.assertEqual(a["blocks"], 4)
        self.assertEqual(a["waste_tokens"], 9)

    def test_contiguous_slab_rule(self):
        # Slab 256: 400 rounds to 512 waste 112; 105 to 256 waste 151
        a = contiguous_allocation(400, slab_tokens=256)
        self.assertEqual(a["allocated_tokens"], 512)
        self.assertEqual(a["waste_tokens"], 112)
        b = contiguous_allocation(105, slab_tokens=256)
        self.assertEqual(b["allocated_tokens"], 256)
        self.assertEqual(b["waste_tokens"], 151)

    def test_contiguous_exact_slab(self):
        a = contiguous_allocation(512, slab_tokens=256)
        self.assertEqual(a["waste_tokens"], 0)
        b = contiguous_allocation(513, slab_tokens=256)
        self.assertEqual(b["allocated_tokens"], 768)
        self.assertEqual(b["waste_tokens"], 255)

    def test_paged_waste_less_than_contiguous_for_small(self):
        paged = paged_allocation(55, block_size=16)
        contig = contiguous_allocation(55, slab_tokens=256)
        self.assertLess(paged["waste_tokens"], contig["waste_tokens"])
        self.assertEqual(contig["waste_tokens"], 201)

    def test_zero_tokens_allocates_nothing(self):
        self.assertEqual(paged_allocation(0)["blocks"], 0)
        self.assertEqual(contiguous_allocation(0)["allocated_tokens"], 0)


class TestSimulation(unittest.TestCase):
    def test_two_requests_hand_computed(self):
        # Outputs 4 and 2, budget 2 per step.
        # Step1: each gets 1 (rem 3,1). Step2: each gets 1 (rem 2,0, B done).
        # Step3: A gets 2 in two passes (rem 0, A done). Total 3 steps.
        records = [RequestRecord("A", 0, 4, 0), RequestRecord("B", 0, 0, 2)]
        sim = simulate_batch(records, tokens_per_step=2)
        self.assertEqual(sim["total_steps"], 3)
        self.assertEqual(sim["per_request_steps"], {"A": 3, "B": 2})
        self.assertEqual(sim["total_output_tokens"], 6)
        self.assertEqual(sim["simulated_throughput_tokens_per_step"], 2.0)

    def test_single_request_budget(self):
        # Output 10, budget 4: steps ceil(10/4)=3, throughput 10/3
        records = [RequestRecord("A", 5, 6, 4)]
        sim = simulate_batch(records, tokens_per_step=4)
        self.assertEqual(sim["total_steps"], 3)
        self.assertEqual(sim["per_request_steps"]["A"], 3)
        self.assertAlmostEqual(
            sim["simulated_throughput_tokens_per_step"], 3.333333)

    def test_zero_output_completes_step_zero(self):
        records = [RequestRecord("A", 10, 0, 0), RequestRecord("B", 0, 3, 0)]
        sim = simulate_batch(records, tokens_per_step=2)
        self.assertEqual(sim["per_request_steps"]["A"], 0)
        # B needs 3 tokens at budget 2: step1 gets 2 (A inactive), step2 gets 1
        self.assertEqual(sim["per_request_steps"]["B"], 2)
        self.assertEqual(sim["total_steps"], 2)

    def test_equal_requests_finish_together(self):
        # Two requests output 4 each, budget 4: each step each gets 2,
        # both finish at step 2.
        records = [RequestRecord("A", 0, 4, 0), RequestRecord("B", 0, 4, 0)]
        sim = simulate_batch(records, tokens_per_step=4)
        self.assertEqual(sim["total_steps"], 2)
        self.assertEqual(sim["per_request_steps"], {"A": 2, "B": 2})

    def test_deterministic_repeat(self):
        records = [RequestRecord("A", 10, 20, 5), RequestRecord("B", 3, 0, 7)]
        first = simulate_batch(records, tokens_per_step=8)
        second = simulate_batch(records, tokens_per_step=8)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
