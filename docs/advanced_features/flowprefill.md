# FlowPrefill

## What it is

FlowPrefill is an experimental scheduler mode for reducing the tail latency impact of long prefills in the unified SGLang engine.

Instead of running a prefill request as one uninterrupted forward pass, FlowPrefill breaks the prefill into multiple **split-prefill** steps. Each step advances a fixed number of transformer layers, then returns control to the scheduler. This gives the scheduler a cooperative preemption point between layer chunks.

When a higher-priority request arrives while a long prefill is in progress, the current split-prefill batch is marked as **preempt pending**. After the current split step finishes, the batch is moved into a preempted queue and can later be resumed from the recorded layer index.

This differs from:

- **Chunked prefill**: splits by prompt tokens.
- **PD disaggregation**: splits prefill and decode onto different workers.
- **Speculative decoding / overlap scheduler**: optimize decode or overlap execution rather than introducing prefill-side cooperative checkpoints.

## How it works

FlowPrefill adds a lightweight lifecycle for prefill requests:

1. A new prefill batch is converted into `SPLIT_PREFILL` mode.
2. Each scheduler iteration runs only `--flowprefill-split-layers` layers.
3. If no higher-priority work arrives, the batch keeps advancing until all hidden layers are processed, then it falls back to normal prefill result handling.
4. If a higher-priority request arrives, the currently running split-prefill batch is marked for cooperative preemption.
5. After the current split step returns, the batch is parked in a preempted queue with its current `split_index`.
6. The scheduler later resumes the parked batch when it outranks the best waiting request.

The current implementation uses the same priority direction as priority scheduling:

- By default, larger integer values mean higher priority.
- If `--schedule-low-priority-values-first` is enabled, smaller integer values mean higher priority.

`priority_fcfs` is the current production policy. `deadline_fcfs` and `slack_edf`
are already accepted by the CLI, and the request path can now carry deadline /
slack metadata into the scheduler. They still do not have a remaining-time
predictor behind them, so they remain experimental. Among requests with the same
effective priority, older requests win.

## When to use it

FlowPrefill is most useful when:

- You serve mixed prompt lengths in a single unified engine.
- Some requests have very long prefills and noticeably delay newer high-priority work.
- You want a scheduler-level latency improvement without moving to PD disaggregation.

It is generally **not** the first feature to enable if your main bottleneck is raw throughput. Start with normal continuous batching, chunked prefill, or PD disaggregation first, then evaluate FlowPrefill specifically for latency-sensitive mixed workloads.

## Requirements and compatibility

FlowPrefill is intentionally narrow in scope right now. It is enabled only when all of the following are true:

- `--enable-flowprefill` is set.
- The model implements `forward_split_prefill`.
- `--flowprefill-granularity layer` is used.
- `--tp-size 1`
- `--pp-size 1`
- DP attention is disabled.
- PD multiplexing is disabled.
- Overlap scheduling is disabled.
- Speculative decoding is disabled.
- PD disaggregation is disabled.
- Chunked prefill is disabled.

If any of these constraints are violated, SGLang will keep serving normally and disable FlowPrefill at scheduler initialization.

## CLI arguments

| Argument | Description | Default |
| --- | --- | --- |
| `--enable-flowprefill` | Enable cooperative preemption for prefills using split-prefill execution. | `False` |
| `--flowprefill-granularity` | Checkpoint granularity. Only `layer` is supported today. | `layer` |
| `--flowprefill-split-layers` | Number of transformer layers executed per split-prefill step. Smaller values create more preemption points but add more scheduler handoffs. | `1` |
| `--flowprefill-max-preemptions` | Maximum cooperative preemptions allowed per request. `0` means unlimited. | `0` |
| `--flowprefill-priority-policy` | Ordering policy for FlowPrefill. `priority_fcfs` is the current production policy. `deadline_fcfs` and `slack_edf` are experimental and currently rely on request-supplied metadata plus simple fallback derivation. | `priority_fcfs` |

## Usage

### Minimal example

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --enable-priority-scheduling \
  --enable-flowprefill \
  --flowprefill-split-layers 2 \
  --flowprefill-max-preemptions 1
```

In this configuration:

- long prefills are advanced two layers at a time;
- a higher-priority request can interrupt the current prefill after the current split step completes;
- each request can be cooperatively preempted at most once.

### With lower numbers meaning higher priority

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --enable-priority-scheduling \
  --schedule-low-priority-values-first \
  --enable-flowprefill \
  --flowprefill-split-layers 1
```

With `--schedule-low-priority-values-first`, smaller priority integers are considered more urgent by both the normal priority scheduler and FlowPrefill.

## Tuning guidance

- Start with `--flowprefill-split-layers 1` for the best responsiveness.
- Increase `--flowprefill-split-layers` if scheduler overhead becomes noticeable.
- Use `--flowprefill-max-preemptions` to prevent very long prompts from being repeatedly pushed back under sustained high-priority traffic.
- Keep `--enable-priority-scheduling` aligned with your client-side priority semantics; FlowPrefill only helps when the scheduler can distinguish urgent requests.
- For the current single-request preemption/resume path, start with
  `--max-running-requests 2` rather than `1`.

## Limitations

- FlowPrefill is currently a **single-engine** optimization, not a replacement for PD disaggregation.
- It only introduces preemption checkpoints between split-prefill steps, not in the middle of a layer chunk.
- Deadline/slack-aware ordering is only partially wired today:
  `deadline_fcfs` and `slack_edf` already accept request metadata through the
  normal request path, but they are not yet backed by a remaining-time predictor
  and therefore are still incomplete.
- It depends on model support for `forward_split_prefill`, so availability varies by model implementation.
- The current resume path is still transitional:
  single-request parked prefills resume from request-owned state, and
  multi-request parked prefills are first split into request-owned per-request
  resume state when it is safe to do so. For parked requests that share the same
  `split_index` and still satisfy the request-owned resume guard, the scheduler
  can now regroup them into a split-prefill batch before resuming. Parked-batch
  resume remains as a compatibility fallback for requests that cannot be sliced
  safely or no longer have resumable request-owned state.
- In the current implementation, do not rely on `--max-running-requests 1`
  for cooperative preemption workloads. In the validated `Qwen3-30B-A3B`
  single-request resume path, the urgent request may still need a fresh req
  slot after the background prefill is parked; using `--max-running-requests 1`
  can therefore fail in `alloc_req_slots()`. Use at least
  `--max-running-requests 2` for this class of workload.
- The request-owned single-request resume path is currently limited to a safe subset.
  Requests using grammar constraints, `input_embeds`, multimodal inputs, or
  encoder-decoder models fall back to parked-batch resume instead of taking the
  lightweight single-request resume path.
- Near-term development is not targeting `input_embeds`, grammar-constrained,
  multimodal, or encoder-decoder request-owned resume support yet; those remain
  future compatibility work.

## Troubleshooting

### FlowPrefill does not seem to activate

Check the scheduler logs first. SGLang will log a warning when FlowPrefill is disabled because of an incompatible feature such as overlap scheduling, PD disaggregation, speculative decoding, TP/PP greater than 1, or missing model support.

### Long prefills still block urgent requests for too long

Reduce `--flowprefill-split-layers` so the scheduler reaches preemption checkpoints more often.

### A long request never seems to finish under constant high-priority traffic

Set `--flowprefill-max-preemptions` to a finite value so a request can only be cooperatively preempted a bounded number of times.
