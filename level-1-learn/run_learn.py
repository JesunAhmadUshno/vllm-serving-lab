"""Run the P03 Learn accounting and simulation. Deterministic.

Usage: python3 level-1-learn/run_learn.py
Writes level-1-learn/results_learn.json and prints the full table.
All results are simulated on CPU. No real model is served.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from serving_sim import (  # noqa: E402
    BLOCK_SIZE_TOKENS,
    CONTIGUOUS_SLAB_TOKENS,
    EXAMPLE_CONFIG,
    TOKENS_PER_STEP,
    RequestRecord,
    contiguous_allocation,
    kv_bytes_for_request,
    kv_bytes_per_token,
    paged_allocation,
    simulate_batch,
)


def load_requests(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [
        RequestRecord(
            request_id=r["request_id"],
            prompt_tokens=int(r["prompt_tokens"]),
            thinking_tokens=int(r["thinking_tokens"]),
            answer_tokens=int(r["answer_tokens"]),
        )
        for r in raw
    ]


def main():
    records = load_requests(os.path.join(HERE, "requests.json"))
    sim = simulate_batch(records, tokens_per_step=TOKENS_PER_STEP)
    per_token = kv_bytes_per_token(EXAMPLE_CONFIG)

    rows = []
    for r in records:
        paged = paged_allocation(r.total_tokens, BLOCK_SIZE_TOKENS)
        contig = contiguous_allocation(r.total_tokens, CONTIGUOUS_SLAB_TOKENS)
        rows.append(
            {
                "request_id": r.request_id,
                "prompt_tokens": r.prompt_tokens,
                "thinking_tokens": r.thinking_tokens,
                "answer_tokens": r.answer_tokens,
                "output_tokens": r.output_tokens,
                "total_tokens": r.total_tokens,
                "kv_bytes": kv_bytes_for_request(r, EXAMPLE_CONFIG),
                "paged_blocks": paged["blocks"],
                "paged_allocated_tokens": paged["allocated_tokens"],
                "paged_waste_tokens": paged["waste_tokens"],
                "contiguous_allocated_tokens": contig["allocated_tokens"],
                "contiguous_waste_tokens": contig["waste_tokens"],
                "steps_to_complete": sim["per_request_steps"][r.request_id],
                "log_line": r.log_line(),
            }
        )

    totals = {
        "requests": len(records),
        "prompt_tokens": sum(r.prompt_tokens for r in records),
        "thinking_tokens": sum(r.thinking_tokens for r in records),
        "answer_tokens": sum(r.answer_tokens for r in records),
        "output_tokens": sum(r.output_tokens for r in records),
        "total_tokens": sum(r.total_tokens for r in records),
        "kv_bytes_total": sum(row["kv_bytes"] for row in rows),
        "paged_allocated_tokens": sum(row["paged_allocated_tokens"] for row in rows),
        "paged_waste_tokens": sum(row["paged_waste_tokens"] for row in rows),
        "contiguous_allocated_tokens": sum(
            row["contiguous_allocated_tokens"] for row in rows
        ),
        "contiguous_waste_tokens": sum(
            row["contiguous_waste_tokens"] for row in rows
        ),
    }

    results = {
        "level": "1-learn",
        "label": "simulated, deterministic, CPU only, no real model served",
        "method": (
            "KV bytes per token = 2 * layers * kv_heads * head_dim * "
            "bytes_per_element. Paged blocks of 16 tokens. Contiguous "
            "baseline rounds up to a fixed 256 token slab. Batching: "
            "round robin passes inside each step, 32 tokens per step "
            "budget, all requests active from step 1, output tokens only."
        ),
        "model_config": {
            "name": EXAMPLE_CONFIG.name,
            "layers": EXAMPLE_CONFIG.layers,
            "kv_heads": EXAMPLE_CONFIG.kv_heads,
            "head_dim": EXAMPLE_CONFIG.head_dim,
            "bytes_per_element": EXAMPLE_CONFIG.bytes_per_element,
            "kv_bytes_per_token": per_token,
        },
        "block_size_tokens": BLOCK_SIZE_TOKENS,
        "contiguous_slab_tokens": CONTIGUOUS_SLAB_TOKENS,
        "simulation": sim,
        "totals": totals,
        "per_request": rows,
    }

    out_path = os.path.join(HERE, "results_learn.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")

    print("P03 Learn: simulated serving accounting (CPU, no real model served)")
    print(f"config={EXAMPLE_CONFIG.name} kv_bytes_per_token={per_token}")
    print(
        "id prompt thinking answer output total kv_bytes paged_blocks "
        "paged_waste contig_waste steps"
    )
    for row in rows:
        print(
            f"{row['request_id']} {row['prompt_tokens']} {row['thinking_tokens']} "
            f"{row['answer_tokens']} {row['output_tokens']} {row['total_tokens']} "
            f"{row['kv_bytes']} {row['paged_blocks']} {row['paged_waste_tokens']} "
            f"{row['contiguous_waste_tokens']} {row['steps_to_complete']}"
        )
        print("  log: " + row["log_line"])
    print(f"totals: {json.dumps(totals, sort_keys=True)}")
    print(
        f"simulation: total_steps={sim['total_steps']} "
        f"total_output_tokens={sim['total_output_tokens']} "
        f"simulated_throughput_tokens_per_step="
        f"{sim['simulated_throughput_tokens_per_step']}"
    )
    print(f"wrote {out_path}")
    return results


if __name__ == "__main__":
    main()
