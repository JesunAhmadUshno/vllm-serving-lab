"""OpenAI-compatible serving API for vllm-serving-lab, Level 2 Build.

Standard library HTTP server (ThreadingHTTPServer) in front of a
pluggable generation backend. Endpoints: GET /health, GET /v1/models,
POST /v1/chat/completions (streaming via server-sent events when
stream is true). Every completion reports usage with thinking tokens
and answer tokens counted apart: the generated text is split at the
"Final answer:" marker and each part is counted by the backend.

FakeBackend is deterministic and exists for tests and CI. It is not a
model and its outputs are never reported as model results.
TransformersBackend loads a real open model on the serving node GPU.

vLLM itself is not used: the fleet GPU node is Windows with no WSL,
and vLLM needs a Linux runtime. The API surface is OpenAI-compatible;
the engine here is Transformers on PyTorch. See PROPOSAL.md, Build spec.
"""
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FINAL_MARKER = "Final answer:"


def split_thinking(text):
    """Split generated text at the final-answer marker.

    Returns (thinking_text, answer_text). When the marker is absent
    the whole text counts as thinking and the answer is empty.
    """
    idx = text.rfind(FINAL_MARKER)
    if idx == -1:
        return text, ""
    return text[:idx], text[idx:]


class FakeBackend:
    """Deterministic backend for tests. Not a model."""

    name = "fake-backend"
    model_id = "fake-0"

    def count_tokens(self, text):
        return len(text.split())

    def generate(self, prompt, max_new_tokens):
        # A fixed reasoning-shaped reply whose length grows with the
        # budget, cut at the budget like a real token cap would cut.
        words = ("step by step: first I restate the problem, then I "
                 "compute each part in order, checking the units as I "
                 "go, and only then do I write the result. " * 8).split()
        budget_words = max(1, max_new_tokens // 2)
        thinking = " ".join(words[:budget_words])
        text = thinking + "\nFinal answer: 42"
        if max_new_tokens < 24:
            text = thinking
        return {
            "text": text,
            "prompt_tokens": self.count_tokens(prompt),
            "completion_tokens": self.count_tokens(text),
        }


class TransformersBackend:
    """Real model backend: Transformers on PyTorch, GPU when present.

    stop_final_answer turns on the serving stop rule: generation
    stops as soon as a complete "Final answer: <number>" line has
    been produced. Without it, this model keeps generating past its
    answer until the token cap, which the Build study measured
    directly (see results_real_nostop.json and the level README).
    """

    def __init__(self, model_id, device="cuda", dtype="float16",
                 stop_final_answer=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = "transformers"
        self.model_id = model_id
        self.stop_final_answer = stop_final_answer
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        torch_dtype = torch.float16 if dtype == "float16" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype).to(device)
        self.device_name = (torch.cuda.get_device_name(0)
                            if device == "cuda" and torch.cuda.is_available()
                            else "cpu")

    def count_tokens(self, text):
        return len(self.tokenizer(text)["input_ids"])

    def _stop_criteria(self, prompt_tokens):
        import re

        from transformers import StoppingCriteria, StoppingCriteriaList

        pattern = re.compile(r"Final answer:\s*-?\d+(?:\.\d+)?\s*\n")
        tokenizer = self.tokenizer

        class StopAtFinalAnswer(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                generated = input_ids[0][prompt_tokens:]
                text = tokenizer.decode(generated, skip_special_tokens=True)
                return bool(pattern.search(text))

        return StoppingCriteriaList([StopAtFinalAnswer()])

    def generate(self, prompt, max_new_tokens):
        torch = self._torch
        messages = [{"role": "user", "content": prompt}]
        text_in = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text_in, return_tensors="pt").to(self.model.device)
        prompt_tokens = inputs["input_ids"].shape[1]
        extra = {}
        if self.stop_final_answer:
            extra["stopping_criteria"] = self._stop_criteria(prompt_tokens)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id, **extra)
        gen = out[0][prompt_tokens:]
        text = self.tokenizer.decode(gen, skip_special_tokens=True)
        return {"text": text, "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(len(gen))}


def completion_payload(backend, prompt, max_new_tokens, model_label):
    started = time.time()
    result = backend.generate(prompt, max_new_tokens)
    elapsed = time.time() - started
    thinking_text, answer_text = split_thinking(result["text"])
    thinking_tokens = backend.count_tokens(thinking_text)
    answer_tokens = result["completion_tokens"] - thinking_tokens
    if answer_tokens < 0:
        answer_tokens = backend.count_tokens(answer_text)
        thinking_tokens = result["completion_tokens"] - answer_tokens
    return {
        "id": "chatcmpl-build",
        "object": "chat.completion",
        "model": model_label,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result["text"]},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
            "thinking_tokens": thinking_tokens,
            "answer_tokens": answer_tokens,
        },
        "elapsed_seconds": round(elapsed, 4),
    }


def make_handler(backend):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _send_json(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send_json(200, {"status": "ok", "backend": backend.name})
            elif self.path == "/v1/models":
                self._send_json(200, {"object": "list", "data": [
                    {"id": backend.model_id, "object": "model"}]})
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self._send_json(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                request = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send_json(400, {"error": "invalid json"})
                return
            messages = request.get("messages") or []
            prompt = "\n".join(m.get("content", "") for m in messages
                               if m.get("role") == "user")
            if not prompt:
                self._send_json(400, {"error": "no user message"})
                return
            max_new = int(request.get("max_tokens") or 128)
            payload = completion_payload(
                backend, prompt, max_new, request.get("model") or backend.model_id)
            if request.get("stream"):
                body_text = payload["choices"][0]["message"]["content"]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                words = body_text.split(" ")
                for i, word in enumerate(words):
                    piece = word if i == 0 else " " + word
                    chunk = {"id": payload["id"], "object": "chat.completion.chunk",
                             "model": payload["model"],
                             "choices": [{"index": 0, "delta": {"content": piece},
                                          "finish_reason": None}]}
                    self._send_sse("data: " + json.dumps(chunk) + "\n\n")
                final = {"id": payload["id"], "object": "chat.completion.chunk",
                         "model": payload["model"], "usage": payload["usage"],
                         "choices": [{"index": 0, "delta": {},
                                      "finish_reason": "stop"}]}
                self._send_sse("data: " + json.dumps(final) + "\n\n")
                self._send_sse("data: [DONE]\n\n")
                self.wfile.write(b"0\r\n\r\n")
            else:
                self._send_json(200, payload)

        def _send_sse(self, text):
            data = text.encode()
            self.wfile.write(("%x\r\n" % len(data)).encode())
            self.wfile.write(data)
            self.wfile.write(b"\r\n")
            self.wfile.flush()

    return Handler


def serve(backend, host="127.0.0.1", port=8000):
    server = ThreadingHTTPServer((host, port), make_handler(backend))
    return server
