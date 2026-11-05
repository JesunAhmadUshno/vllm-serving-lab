"""Deterministic serving simulator and token accounting for P03 Learn.

No real model is served. No GPU is used. All numbers are simulated,
deterministic, and labelled as simulated. Method is in the README.

Stdlib only.
"""
from dataclasses import dataclass


BLOCK_SIZE_TOKENS = 16
CONTIGUOUS_SLAB_TOKENS = 256
TOKENS_PER_STEP = 32


@dataclass(frozen=True)
class ModelConfig:
    """Named example config for KV cache math."""

    name: str = "example-32L-8KV-128D-fp16"
    layers: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    bytes_per_element: int = 2


EXAMPLE_CONFIG = ModelConfig()


@dataclass(frozen=True)
class RequestRecord:
    """One request with thinking tokens logged apart from answer tokens."""

    request_id: str
    prompt_tokens: int
    thinking_tokens: int
    answer_tokens: int

    @property
    def total_tokens(self):
        return self.prompt_tokens + self.thinking_tokens + self.answer_tokens

    @property
    def output_tokens(self):
        return self.thinking_tokens + self.answer_tokens

    @property
    def is_reasoning(self):
        return self.thinking_tokens > 0

    def log_line(self):
        return (
            f"request={self.request_id} prompt={self.prompt_tokens} "
            f"thinking={self.thinking_tokens} answer={self.answer_tokens} "
            f"output={self.output_tokens} total={self.total_tokens} "
            f"reasoning={str(self.is_reasoning).lower()}"
        )


def kv_bytes_per_token(config=EXAMPLE_CONFIG):
    """Bytes per token = 2 (K and V) * layers * kv_heads * head_dim * bytes."""
    return (
        2
        * config.layers
        * config.kv_heads
        * config.head_dim
        * config.bytes_per_element
    )


def kv_bytes_for_tokens(n_tokens, config=EXAMPLE_CONFIG):
    return n_tokens * kv_bytes_per_token(config)


def kv_bytes_for_request(record, config=EXAMPLE_CONFIG):
    return kv_bytes_for_tokens(record.total_tokens, config)


def paged_allocation(total_tokens, block_size=BLOCK_SIZE_TOKENS):
    """PagedAttention style: allocate whole blocks of block_size tokens."""
    if total_tokens <= 0:
        return {"blocks": 0, "allocated_tokens": 0, "waste_tokens": 0}
    blocks = -(-total_tokens // block_size)
    allocated = blocks * block_size
    return {
        "blocks": blocks,
        "allocated_tokens": allocated,
        "waste_tokens": allocated - total_tokens,
    }


def contiguous_allocation(total_tokens, slab_tokens=CONTIGUOUS_SLAB_TOKENS):
    """Contiguous baseline: round up to a fixed slab of slab_tokens.

    Slab rule: every request gets the next whole multiple of 256 tokens,
    however small the request is. That is the waste paged blocks reduce.
    """
    if total_tokens <= 0:
        return {"slabs": 0, "allocated_tokens": 0, "waste_tokens": 0}
    slabs = -(-total_tokens // slab_tokens)
    allocated = slabs * slab_tokens
    return {
        "slabs": slabs,
        "allocated_tokens": allocated,
        "waste_tokens": allocated - total_tokens,
    }


def simulate_batch(records, tokens_per_step=TOKENS_PER_STEP):
    """Deterministic token-step batching simulation.

    All records are active from step 1. In each step the engine has a
    budget of tokens_per_step generation tokens. Inside a step it makes
    repeated round robin passes in the given order, giving 1 token to
    each active request per pass, until the budget is used or every
    active request finishes. A request that needs 0 output tokens
    completes at step 0. Only output tokens (thinking plus answer) are
    generated in the simulation. Prompt tokens are prefill and are
    counted in KV and block accounting, not in generation steps.

    Throughput is tokens per step, never tokens per second. No wall
    time is measured or claimed.
    """
    remaining = {r.request_id: r.output_tokens for r in records}
    completion_step = {}
    for r in records:
        if r.output_tokens == 0:
            completion_step[r.request_id] = 0
    step = 0
    while any(v > 0 for v in remaining.values()):
        step += 1
        budget = tokens_per_step
        while budget > 0 and any(v > 0 for v in remaining.values()):
            for r in records:
                if budget <= 0:
                    break
                if remaining[r.request_id] > 0:
                    remaining[r.request_id] -= 1
                    budget -= 1
                    if remaining[r.request_id] == 0:
                        completion_step[r.request_id] = step
    total_output = sum(r.output_tokens for r in records)
    throughput = (total_output / step) if step else 0.0
    return {
        "tokens_per_step": tokens_per_step,
        "total_steps": step,
        "total_output_tokens": total_output,
        "simulated_throughput_tokens_per_step": round(throughput, 6),
        "per_request_steps": {r.request_id: completion_step[r.request_id] for r in records},
    }
