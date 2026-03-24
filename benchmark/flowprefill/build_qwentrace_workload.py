#!/usr/bin/env python3
"""
Build a FlowPrefill-friendly workload from the public Bailian trace.

The paper "FlowPrefill" states that QwenTrace is preprocessed into
single-turn queries before evaluation. The public `qwen_traceA_blksz_16.jsonl`
still contains multi-turn chains, so this script removes inherited history by
subtracting the longest shared block prefix with the parent request.

Output format: one JSON object per line.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


SLO_PRESETS_MS = {
    "llama3-8b": {"text": 250.0, "image": 500.0, "search": 4000.0, "file": 6000.0},
    "qwen2.5-14b": {"text": 400.0, "image": 800.0, "search": 6500.0, "file": 9000.0},
    "llama3-70b": {"text": 1000.0, "image": 2000.0, "search": 15000.0, "file": 18000.0},
}

DEFAULT_PRIORITY = {"text": 4, "image": 3, "search": 2, "file": 1}
BLOCK_SIZE = 16


@dataclass
class TraceItem:
    chat_id: int
    parent_chat_id: int
    timestamp: float
    input_length: int
    output_length: int
    task_type: str
    turn: int
    hash_ids: List[int]

    @classmethod
    def from_json(cls, obj: Dict) -> "TraceItem":
        return cls(
            chat_id=obj["chat_id"],
            parent_chat_id=obj["parent_chat_id"],
            timestamp=obj["timestamp"],
            input_length=obj["input_length"],
            output_length=obj["output_length"],
            task_type=obj["type"],
            turn=obj["turn"],
            hash_ids=obj["hash_ids"],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="qwen-bailian-usagetraces-anon/qwen_traceA_blksz_16.jsonl",
        help="Path to the public Qwen/Bailian trace. Use traceA for the 4-task To-C workload.",
    )
    parser.add_argument(
        "--output",
        default="sglang/benchmark/flowprefill/qwentrace_flowprefill_qwen2.5-14b.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--model-preset",
        default="qwen2.5-14b",
        choices=sorted(SLO_PRESETS_MS),
        help="Paper Table 2 SLO preset.",
    )
    parser.add_argument(
        "--slo-scale",
        type=float,
        default=1.0,
        help="Multiply all TTFT SLOs by this factor. Paper Figure 9 sweeps this value.",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="Divide inter-arrival time by this factor. 2.0 replays at 2x request rate.",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Optional cap after sorting by timestamp.",
    )
    return parser.parse_args()


def load_trace(path: Path) -> List[TraceItem]:
    items: List[TraceItem] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(TraceItem.from_json(json.loads(line)))
    items.sort(key=lambda x: x.timestamp)
    return items


def shared_prefix_blocks(cur: TraceItem, parent: Optional[TraceItem]) -> int:
    if parent is None:
        return 0
    n = min(len(cur.hash_ids), len(parent.hash_ids))
    i = 0
    while i < n and cur.hash_ids[i] == parent.hash_ids[i]:
        i += 1
    return i


def single_turn_prompt_len(cur: TraceItem, parent: Optional[TraceItem]) -> int:
    prefix_blocks = shared_prefix_blocks(cur, parent)
    prefix_tokens = min(prefix_blocks * BLOCK_SIZE, cur.input_length)
    return max(1, cur.input_length - prefix_tokens)


def output_record(
    idx: int,
    item: TraceItem,
    parent: Optional[TraceItem],
    first_ts: float,
    prev_scaled_ts: float,
    slo_map: Dict[str, float],
    time_scale: float,
) -> Dict:
    scaled_ts = (item.timestamp - first_ts) / time_scale
    prompt_len = single_turn_prompt_len(item, parent)
    task_slo_ms = slo_map[item.task_type]
    prefix_blocks = shared_prefix_blocks(item, parent)
    return {
        "request_id": idx,
        "chat_id": item.chat_id,
        "parent_chat_id": item.parent_chat_id,
        "turn": item.turn,
        "task_type": item.task_type,
        "timestamp_s": scaled_ts,
        "arrival_delay_s": max(0.0, scaled_ts - prev_scaled_ts),
        "prompt_len": prompt_len,
        "full_prompt_len": item.input_length,
        "output_len": item.output_length,
        "shared_prefix_blocks_with_parent": prefix_blocks,
        "shared_prefix_tokens_with_parent": prefix_blocks * BLOCK_SIZE,
        "ttft_slo_ms": task_slo_ms,
        "priority": DEFAULT_PRIORITY[item.task_type],
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    items = load_trace(input_path)
    if args.max_requests is not None:
        items = items[: args.max_requests]

    item_by_chat_id = {item.chat_id: item for item in items}
    base_slos = SLO_PRESETS_MS[args.model_preset]
    slo_map = {k: v * args.slo_scale for k, v in base_slos.items()}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_ts = items[0].timestamp
    prev_scaled_ts = 0.0

    with output_path.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(items):
            parent = item_by_chat_id.get(item.parent_chat_id)
            record = output_record(
                idx=idx,
                item=item,
                parent=parent,
                first_ts=first_ts,
                prev_scaled_ts=prev_scaled_ts,
                slo_map=slo_map,
                time_scale=args.time_scale,
            )
            prev_scaled_ts = record["timestamp_s"]
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    print(f"Wrote {len(items)} requests to {output_path}")


if __name__ == "__main__":
    main()
