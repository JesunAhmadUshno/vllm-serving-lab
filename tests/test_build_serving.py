"""Build tests: serving API shape, study scoring, load test math."""
import json
import os
import sys
import threading
import unittest
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "level-2-build"))
sys.path.insert(0, os.path.join(ROOT, "level-1-learn"))

import load_test  # noqa: E402
import server as server_mod  # noqa: E402
import study  # noqa: E402


def post(base, path, payload):
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=10) as r:
        return json.loads(r.read())


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = server_mod.serve(server_mod.FakeBackend(), port=0)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = "http://127.0.0.1:%d" % cls.port

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def test_health(self):
        self.assertEqual(get(self.base, "/health")["status"], "ok")

    def test_models(self):
        data = get(self.base, "/v1/models")["data"]
        self.assertEqual(data[0]["id"], "fake-0")

    def test_completion_shape_and_usage(self):
        payload = post(self.base, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "What is 6 * 7?"}],
            "max_tokens": 128})
        usage = payload["usage"]
        self.assertEqual(usage["total_tokens"],
                         usage["prompt_tokens"] + usage["completion_tokens"])
        self.assertEqual(usage["thinking_tokens"] + usage["answer_tokens"],
                         usage["completion_tokens"])
        self.assertIn("Final answer:", payload["choices"][0]["message"]["content"])

    def test_thinking_split_counts(self):
        payload = server_mod.completion_payload(
            server_mod.FakeBackend(), "q", 128, "fake-0")
        self.assertGreater(payload["usage"]["thinking_tokens"], 0)
        self.assertGreater(payload["usage"]["answer_tokens"], 0)

    def test_streaming_chunks_reassemble(self):
        req = urllib.request.Request(
            self.base + "/v1/chat/completions",
            data=json.dumps({
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 64, "stream": True}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            raw = r.read().decode()
        pieces = []
        usage = None
        for line in raw.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line[6:])
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta:
                    pieces.append(delta["content"])
                if "usage" in chunk:
                    usage = chunk["usage"]
        whole = post(self.base, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 64})
        self.assertEqual("".join(pieces),
                         whole["choices"][0]["message"]["content"])
        self.assertIsNotNone(usage)

    def test_missing_user_message_400(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            post(self.base, "/v1/chat/completions", {"messages": []})
        self.assertEqual(ctx.exception.code, 400)


class StudyTests(unittest.TestCase):
    def test_extract_prefers_final_answer(self):
        self.assertEqual(study.extract_answer("work 99 then Final answer: 42"), 42.0)

    def test_extract_falls_back_to_last_number(self):
        self.assertEqual(study.extract_answer("first 3 then 7"), 7.0)

    def test_extract_none(self):
        self.assertIsNone(study.extract_answer("no numbers here"))

    def test_extract_decimal_and_negative(self):
        self.assertEqual(study.extract_answer("Final answer: -3.5"), -3.5)

    def test_is_correct_tolerance(self):
        self.assertTrue(study.is_correct(74.8, 74.80))
        self.assertFalse(study.is_correct(75.0, 74.80))
        self.assertFalse(study.is_correct(None, 1))

    def test_questions_schema(self):
        questions = study.load_questions()
        self.assertEqual(len(questions), 10)
        for q in questions:
            self.assertIn("question", q)
            self.assertIsInstance(q["expected"], (int, float))

    def test_run_study_fake_counts(self):
        questions = study.load_questions()
        out = study.run_study(server_mod.FakeBackend(), questions, [128])
        budget = out["budgets"]["128"]
        self.assertEqual(budget["questions"], 10)
        # The fake backend always ends with 42, and exactly one
        # question (q07) expects 42.
        self.assertEqual(budget["correct"], 1)
        self.assertEqual(budget["tokens_per_correct"],
                         round(budget["completion_tokens"] / 1, 2))
        self.assertEqual(len(out["rows"]), 10)


class LoadTestMathTests(unittest.TestCase):
    def test_percentile(self):
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.assertEqual(load_test.percentile(values, 50), 5)
        self.assertEqual(load_test.percentile(values, 95), 10)
        self.assertIsNone(load_test.percentile([], 50))

    def test_aggregate(self):
        results = [
            {"ok": True, "latency_seconds": 0.1, "completion_tokens": 50},
            {"ok": True, "latency_seconds": 0.3, "completion_tokens": 70},
            {"ok": False, "latency_seconds": 0.2},
        ]
        summary = load_test.aggregate(results, 2.0)
        self.assertEqual(summary["succeeded"], 2)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["requests_per_second"], 1.0)
        self.assertEqual(summary["completion_tokens_per_second"], 60.0)


if __name__ == "__main__":
    unittest.main()
