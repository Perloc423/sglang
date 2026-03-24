#!/usr/bin/env python3
"""
Replay a prebuilt FlowPrefill workload JSONL against the native /generate API.

Expected workload format is the output of:
  sglang/benchmark/flowprefill/build_qwentrace_workload.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp
from transformers import AutoTokenizer

BENCH_AIOHTTP_TIMEOUT_SECONDS = 6 * 60 * 60
BENCH_AIOHTTP_READ_BUFSIZE_BYTES = 10 * 1024 * 1024


@dataclass
class ReplayRequest:
    request_id: int
    task_type: str
    prompt_len: int
    output_len: int
    priority: int
    arrival_delay_s: float
    timestamp_s: float
    ttft_slo_ms: Optional[float]
    prompt: str


@dataclass
class ReplayResult:
    request_id: int
    task_type: str
    prompt_len: int
    output_len: int
    priority: int
    arrival_delay_s: float
    ttft_slo_ms: Optional[float]
    ttft_ms: float = 0.0
    latency_ms: float = 0.0
    slo_met: Optional[bool] = None
    success: bool = False
    error: str = ""
    output_text: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--model", required=True, help="Served model id or local model path.")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="HF tokenizer path. Defaults to --model when omitted.",
    )
    parser.add_argument("--workload-file", required=True, help="Input workload JSONL.")
    parser.add_argument("--output-file", required=True, help="Result JSON.")
    parser.add_argument(
        "--request-rate-scale",
        type=float,
        default=1.0,
        help="Replay rate multiplier. 2.0 means send requests twice as fast.",
    )
    parser.add_argument(
        "--window-start-seconds",
        type=float,
        default=0.0,
        help="Start offset in seconds on the workload timeline.",
    )
    parser.add_argument(
        "--time-window-seconds",
        type=float,
        default=None,
        help="Replay only requests inside the selected contiguous time window.",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Optional cap after applying the time window.",
    )
    parser.add_argument(
        "--print-window-summary",
        action="store_true",
        default=False,
        help="Print a short summary of the selected contiguous time window before replay.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        default=False,
        help="Set sampling_params.ignore_eos=true.",
    )
    parser.add_argument(
        "--override-max-new-tokens",
        type=int,
        default=None,
        help="If set, override every request's output_len/max_new_tokens with this value.",
    )
    return parser.parse_args()


def create_session() -> aiohttp.ClientSession:
    timeout = aiohttp.ClientTimeout(total=BENCH_AIOHTTP_TIMEOUT_SECONDS)
    return aiohttp.ClientSession(
        timeout=timeout, read_bufsize=BENCH_AIOHTTP_READ_BUFSIZE_BYTES
    )


def encode_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def build_prompt_with_exact_tokens(tokenizer, target_len: int, prefix: str) -> tuple[str, int]:
    prefix_len = encode_len(tokenizer, prefix)
    if prefix_len > target_len:
        raise ValueError(
            f"Prompt prefix already uses {prefix_len} tokens, exceeding target_len={target_len}."
        )

    prompt = prefix + (" a" * max(target_len - prefix_len, 0))
    real_len = encode_len(tokenizer, prompt)
    if real_len < target_len:
        prompt += " test" * (target_len - real_len)
        real_len = encode_len(tokenizer, prompt)

    if real_len > target_len:
        ids = tokenizer.encode(prompt, add_special_tokens=False)[:target_len]
        prompt = tokenizer.decode(ids, skip_special_tokens=False)
        real_len = encode_len(tokenizer, prompt)
        if real_len != target_len:
            raise ValueError(
                f"Failed to build exact-length prompt for target_len={target_len}, got {real_len}."
            )

    return prompt, real_len


def build_prompt_for_request(tokenizer, request_id: int, task_type: str, target_len: int) -> tuple[str, int]:
    prefix_candidates = [
        f"flowprefill_trace_request_id={request_id}\ntask_type={task_type}\n",
        f"req={request_id} type={task_type}\n",
        f"{task_type}:{request_id}\n",
        f"{request_id}\n",
        "",
    ]

    for prefix in prefix_candidates:
        if encode_len(tokenizer, prefix) <= target_len:
            return build_prompt_with_exact_tokens(tokenizer, target_len, prefix)

    raise ValueError(f"Failed to build prompt for request_id={request_id}, target_len={target_len}.")


def load_workload(
    path: Path,
    tokenizer,
    window_start_seconds: float,
    time_window_seconds: Optional[float],
    max_requests: Optional[int],
    override_max_new_tokens: Optional[int],
) -> List[ReplayRequest]:
    requests: List[ReplayRequest] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            prompt, real_len = build_prompt_for_request(
                tokenizer=tokenizer,
                request_id=int(obj["request_id"]),
                task_type=obj["task_type"],
                target_len=int(obj["prompt_len"]),
            )
            requests.append(
                ReplayRequest(
                    request_id=int(obj["request_id"]),
                    task_type=obj["task_type"],
                    prompt_len=real_len,
                    output_len=(
                        int(override_max_new_tokens)
                        if override_max_new_tokens is not None
                        else int(obj["output_len"])
                    ),
                    priority=int(obj["priority"]),
                    arrival_delay_s=float(obj["arrival_delay_s"]),
                    timestamp_s=float(obj.get("timestamp_s", 0.0)),
                    ttft_slo_ms=(
                        float(obj["ttft_slo_ms"])
                        if obj.get("ttft_slo_ms") is not None
                        else None
                    ),
                    prompt=prompt,
                )
            )

    requests.sort(key=lambda req: req.timestamp_s)

    window_end_seconds = (
        None
        if time_window_seconds is None
        else window_start_seconds + time_window_seconds
    )
    requests = [
        req
        for req in requests
        if req.timestamp_s >= window_start_seconds
        and (window_end_seconds is None or req.timestamp_s < window_end_seconds)
    ]

    if max_requests is not None:
        requests = requests[:max_requests]

    if requests:
        first_timestamp_s = requests[0].timestamp_s
        prev_timestamp_s: Optional[float] = None
        for req in requests:
            req.timestamp_s = req.timestamp_s - first_timestamp_s
            if prev_timestamp_s is None:
                req.arrival_delay_s = 0.0
            else:
                req.arrival_delay_s = max(0.0, req.timestamp_s - prev_timestamp_s)
            prev_timestamp_s = req.timestamp_s

    return requests


def summarize_selected_window(requests: List[ReplayRequest]) -> Dict:
    by_type: Dict[str, int] = {}
    for req in requests:
        by_type[req.task_type] = by_type.get(req.task_type, 0) + 1

    if not requests:
        return {"count": 0, "duration_s": 0.0, "avg_rps": 0.0, "by_type": by_type}

    duration_s = requests[-1].timestamp_s - requests[0].timestamp_s
    avg_rps = (len(requests) / duration_s) if duration_s > 0 else float(len(requests))
    return {
        "count": len(requests),
        "duration_s": duration_s,
        "avg_rps": avg_rps,
        "by_type": by_type,
    }


async def send_one(
    session: aiohttp.ClientSession,
    api_url: str,
    req: ReplayRequest,
    model: str,
    ignore_eos: bool,
) -> ReplayResult:
    result = ReplayResult(
        request_id=req.request_id,
        task_type=req.task_type,
        prompt_len=req.prompt_len,
        output_len=req.output_len,
        priority=req.priority,
        arrival_delay_s=req.arrival_delay_s,
        ttft_slo_ms=req.ttft_slo_ms,
    )

    payload = {
        "text": req.prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": req.output_len,
            "ignore_eos": ignore_eos,
        },
        "stream": True,
        "priority": req.priority,
        "model": model,
    }
    if req.ttft_slo_ms is not None:
        payload["prefill_ttft_slo_ms"] = req.ttft_slo_ms

    st = time.perf_counter()
    first_token_ts: Optional[float] = None
    output_fragments: List[str] = []

    try:
        async with session.post(api_url, json=payload) as resp:
            if resp.status != 200:
                result.error = f"HTTP {resp.status}: {await resp.text()}"
                return result

            async for chunk_bytes in resp.content:
                chunk_bytes = chunk_bytes.strip()
                if not chunk_bytes:
                    continue
                now = time.perf_counter()
                if first_token_ts is None:
                    first_token_ts = now

                chunk = chunk_bytes.decode("utf-8")
                if chunk.startswith("data: "):
                    chunk = chunk[6:]
                if chunk == "[DONE]":
                    continue
                try:
                    obj = json.loads(chunk)
                except json.JSONDecodeError:
                    continue

                text = obj.get("text")
                if isinstance(text, str):
                    output_fragments.append(text)

        end_ts = time.perf_counter()
        result.ttft_ms = ((first_token_ts or end_ts) - st) * 1000.0
        result.latency_ms = (end_ts - st) * 1000.0
        result.output_text = "".join(output_fragments)
        result.success = True
        if result.ttft_slo_ms is not None:
            result.slo_met = result.ttft_ms <= result.ttft_slo_ms
        return result
    except Exception as exc:
        result.error = repr(exc)
        return result


async def replay_requests(
    requests: List[ReplayRequest],
    api_url: str,
    model: str,
    ignore_eos: bool,
    request_rate_scale: float,
) -> List[ReplayResult]:
    results: List[Optional[ReplayResult]] = [None] * len(requests)
    async with create_session() as session:
        start = time.perf_counter()

        async def launch(i: int, req: ReplayRequest) -> None:
            target = req.arrival_delay_s / request_rate_scale
            wait = max(0.0, start + target - time.perf_counter())
            if wait > 0:
                await asyncio.sleep(wait)
            results[i] = await send_one(
                session=session,
                api_url=api_url,
                req=req,
                model=model,
                ignore_eos=ignore_eos,
            )

        await asyncio.gather(*(launch(i, req) for i, req in enumerate(requests)))

    return [x for x in results if x is not None]


def summarize(results: List[ReplayResult]) -> Dict:
    groups: Dict[str, Dict] = {}
    total = len(results)
    successful = sum(1 for r in results if r.success)

    for task_type in sorted({r.task_type for r in results}):
        rs = [r for r in results if r.task_type == task_type]
        ttfts = sorted(r.ttft_ms for r in rs if r.success)
        slo_checked = sum(1 for r in rs if r.ttft_slo_ms is not None and r.success)
        slo_met = sum(1 for r in rs if r.slo_met)
        groups[task_type] = {
            "count": len(rs),
            "successful": sum(1 for r in rs if r.success),
            "ttft_p50_ms": ttfts[len(ttfts) // 2] if ttfts else None,
            "ttft_p99_ms": ttfts[min(len(ttfts) - 1, int(len(ttfts) * 0.99))]
            if ttfts
            else None,
            "slo_checked": slo_checked,
            "slo_met": slo_met,
            "slo_attainment_pct": (100.0 * slo_met / slo_checked) if slo_checked else None,
            "errors": [r.error for r in rs if r.error][:10],
        }

    overall_slo_checked = sum(1 for r in results if r.ttft_slo_ms is not None and r.success)
    overall_slo_met = sum(1 for r in results if r.slo_met)
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "overall": {
            "total": total,
            "successful": successful,
            "failed": total - successful,
            "slo_checked": overall_slo_checked,
            "slo_met": overall_slo_met,
            "slo_attainment_pct": (
                100.0 * overall_slo_met / overall_slo_checked
                if overall_slo_checked
                else None
            ),
        },
        "groups": groups,
        "per_request": [r.__dict__ for r in results],
    }


async def amain() -> None:
    args = parse_args()
    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    requests = load_workload(
        path=Path(args.workload_file),
        tokenizer=tokenizer,
        window_start_seconds=args.window_start_seconds,
        time_window_seconds=args.time_window_seconds,
        max_requests=args.max_requests,
        override_max_new_tokens=args.override_max_new_tokens,
    )
    if args.print_window_summary:
        print(json.dumps(summarize_selected_window(requests), ensure_ascii=True))
    api_url = f"http://{args.host}:{args.port}/generate"

    results = await replay_requests(
        requests=requests,
        api_url=api_url,
        model=args.model,
        ignore_eos=args.ignore_eos,
        request_rate_scale=args.request_rate_scale,
    )
    summary = summarize(results)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    print(
        json.dumps(
            {
                "total": summary["overall"]["total"],
                "successful": summary["overall"]["successful"],
                "slo_attainment_pct": summary["overall"]["slo_attainment_pct"],
                "output_file": str(output_path),
            },
            ensure_ascii=True,
        )
    )


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
