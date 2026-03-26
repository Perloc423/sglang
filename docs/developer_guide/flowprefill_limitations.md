# FlowPrefill Current Limitations and Mitigations

This note summarizes the current limitations observed from the local
`QwenTrace` replay experiments under `slack_edf`, and lists practical
mitigations for the current SGLang implementation.

The goal here is not to restate the FlowPrefill paper. The goal is to record
what the current code does, where it can behave poorly, and what to try next.

## Scope

The observations below are based on:

- mixed `text` / `image` / `search` / `file` workloads derived from
  `qwen_traceA_blksz_16.jsonl`
- request-level TTFT SLO metadata
- `slack_edf` as the FlowPrefill priority policy
- a replay setup that can force `max_new_tokens=1` to isolate TTFT/prefill
  effects from long decode tails

These findings are therefore about the current SGLang implementation and
benchmark setup, not a claim about the final FlowPrefill design space.

The current code is already more advanced than a plain EDF baseline. In
particular, `slack_edf` now includes:

- feasible-first ordering
- prompt-length bucketized runtime estimation
- waiting-side candidate batching
- candidate harm / protected-harm penalties
- regrouped resume for parked requests with the same `split_index`

These mechanisms help, but they do not remove the limitations described below.

Current implementation notes worth calling out explicitly:

- request-owned resume and same-`split_index` regroup are already integrated
- `slack_edf` now uses feasible-first ordering across waiting and parked work
- non-short requests have a bounded rescue path after extended waiting
- repeatedly preempted non-short requests get a short preemption cooldown after
  resume
- a later attempt to cap waiting-side medium/long batch size was benchmarked
  and reverted because it worsened both tail latency and class-level SLO
  attainment

## Observed limitations

### 1. `slack_edf` is sensitive to predictor error

The current policy computes:

```text
slack = deadline - now - TTFT_hat
```

If `TTFT_hat` is wrong, request ordering can be wrong even if the scheduler
mechanics are functioning correctly.

In the current implementation, `TTFT_hat` is still a lightweight heuristic
based on observed split-prefill runtime. That means:

- requests can be marked infeasible too early
- requests can also be considered feasible when they are already unlikely to
  meet SLO
- the error matters most for tight-SLO short requests

Practical effect:

- a short request with a strict SLO can be demoted as soon as its slack becomes
  slightly negative
- a looser-SLO long request can remain feasible for much longer and keep
  winning scheduling decisions

### 2. Short tight-SLO requests can be hard to recover once classified as infeasible

This is the clearest limitation exposed by the current replay.

When the system is under overload, once a short request is assigned negative
slack, `slack_edf` has limited mechanisms to "rescue" it later. The request is
already outside the feasible set, so the scheduler tends to prefer requests
that still have positive slack.

Practical effect:

- `text` requests with very small TTFT budgets are often sacrificed first
- `search` / `file` requests with much looser SLOs can achieve better goodput
  even when they are longer
- this can improve aggregate SLO attainment while making interactive traffic
  worse

This is not a correctness bug. It is a policy limitation under overload.

### 3. FlowPrefill only helps with prefill-side head-of-line blocking

FlowPrefill creates cooperative preemption points during prefill. It does not
solve long decode tails.

If the workload is decode-bound, FlowPrefill can add scheduling overhead
without changing the dominant bottleneck.

Practical effect:

- benchmark conclusions can be badly distorted if long generations are left
  enabled
- in mixed workloads, the replay harness should support forcing
  `max_new_tokens=1` when the goal is TTFT analysis

### 4. Aggregate SLO can improve while the most important class gets worse

Under mixed SLO workloads, `slack_edf` can trade one class against another.

For example:

- the scheduler may improve `search` / `file` goodput
- at the same time, `text` TTFT can regress because many `text` requests are
  deemed infeasible early

This means "overall SLO attainment improved" is not enough by itself. Results
must be broken down by request class.

This behavior was observed directly in local replay:

- aggregate SLO attainment improved under `max_new_tokens=1`
- but `text` still regressed while `search` / `file` improved

That is a real product tradeoff, not just a reporting artifact.

### 5. Overload pushes the system into a regime where policy differences become hard to interpret

If the majority of requests are already infeasible on arrival, policy behavior
is dominated by triage rather than optimization.

In that regime:

- a scheduler may appear better only because it gives up on one class faster
- predictor noise is amplified
- preemption and resume costs are more likely to show up as pure overhead

This makes benchmark calibration important. A deeply overloaded trace is still
useful, but it should not be the only evaluation point.

### 6. Current bounded rescue / cooldown logic is a stabilization layer, not a final solution

Recent scheduler changes added two targeted mitigations for non-short requests:

- bounded long-request rescue after sufficiently long waiting
- a short preemption cooldown for repeatedly preempted non-short requests

These changes fixed one concrete failure mode: large rescue batches no longer
take over the main scheduling path as aggressively as before. However, replay
results still show:

- `search` tail latency can remain multi-second or even tens of seconds under
  sustained overload
- `text` can still suffer large p99 spikes in the tightest SLO region
- improving class-level SLO attainment does not imply healthy tail behavior

So these mechanisms should be understood as guardrails. They do not yet solve
the deeper tension between:

- protecting tight-SLO short requests
- preserving acceptable tail latency for long loose-SLO requests

### 7. The reverted waiting-candidate batch-size cap is a documented dead end

A later experiment added a hard cap to waiting-side `slack_edf` batching:

- medium candidate size `<= 2`
- long candidate size `<= 1`

This looked attractive because logs suggested large same-bucket batches were
occupying too much scheduler time. In practice it made things worse:

- requests became more fragmented
- rescue fired more often
- `file/search` SLO attainment regressed
- tail latency became even larger

That cap has been reverted. It is useful to record this explicitly so future
work does not re-run the same experiment without a stronger justification.

## Recommended mitigations

### 1. Always evaluate TTFT with a decode-controlled benchmark

When the question is "does FlowPrefill help prefill scheduling?", decode must
be controlled first.

Recommended practice:

- run one benchmark with the workload's original `output_len`
- run a second benchmark with `max_new_tokens=1`
- compare conclusions before blaming or praising the scheduler

This separates:

- prefill scheduling effects
- decode-tail effects

### 2. Treat per-request SLO classes as mandatory input, not optional metadata

`slack_edf` behaves poorly when mixed workloads are forced under one uniform
server-level default SLO.

Recommended practice:

- pass request-specific `prefill_ttft_slo_ms` whenever possible
- at minimum, keep separate SLO classes for `text`, `image`, `search`, and
  `file`
- do not rely on one global default for mixed workloads

### 3. Add an anti-starvation or rescue path for recently-infeasible short requests

The current feasible-first behavior is simple and understandable, but it can
freeze out urgent short requests once they become slightly negative in slack.

Potential mitigations:

- add a bounded "rescue window" for short requests whose slack just crossed
  below zero
- cap how aggressively infeasible requests are demoted
- introduce a separate boost for short tight-SLO requests after bounded waiting
- combine slack with age so recently arrived urgent requests are not dropped too
  early

One concrete direction:

```text
effective_priority = class_bias + feasibility_term + age_term + rescue_bonus
```

where `rescue_bonus` only applies to a small short-request subset and is
bounded in time.

### 4. Improve the remaining-time predictor before drawing strong policy conclusions

Current `slack_edf` quality is tightly coupled to `TTFT_hat`.

Potential improvements:

- stronger prompt-length bucketing
- batch-shape-aware runtime modeling
- EMA instead of raw local averages
- separate models for cold-start vs warmed-up resumed requests
- explicit handling for requests that just crossed from feasible to infeasible
- explicit calibration against benchmark traces

Without better prediction, scheduler comparisons can mostly reflect estimator
quality rather than policy quality.

### 5. Tune for the intended product objective, not only overall goodput

If the product objective is interactive chat quality, then `text` should not be
quietly sacrificed to improve looser-SLO classes.

Recommended practice:

- report metrics by request class
- define a primary target metric for `text`
- treat aggregate goodput as secondary when class priorities are asymmetric

Possible policy extensions:

- class-aware feasibility ordering
- minimum service guarantees for `text`
- explicit "protect urgent interactive traffic first" mode

### 6. Benchmark across multiple load regions

Current results should be collected in at least three regions:

- underloaded: most requests feasible
- moderately loaded: policy choice matters most
- overloaded: triage behavior dominates

If FlowPrefill is only tested in deep overload, the results mainly describe
failure behavior rather than steady-state scheduler quality.

### 7. Keep the implementation path simple while validating policy ideas

When evaluating `slack_edf`, reduce unrelated moving parts first.

Recommended validation order:

1. force `max_new_tokens=1`
2. fix one contiguous time window from the trace
3. compare `priority_fcfs` vs `deadline_fcfs` vs `slack_edf`
4. inspect class-level results
5. only then reintroduce full output lengths

This prevents decode tails and workload drift from masking scheduler behavior.

Recommended local workflow:

1. build a workload with
   [`build_qwentrace_workload.py`](/share/wwmq/mywork/sglang/benchmark/flowprefill/build_qwentrace_workload.py)
2. replay it with
   [`bench_flowprefill_trace_replay.py`](/share/wwmq/mywork/sglang/benchmark/flowprefill/bench_flowprefill_trace_replay.py)
3. select a contiguous replay window with `--window-start-seconds` and
   `--time-window-seconds`
4. first run with `--override-max-new-tokens 1`
5. then rerun with original output lengths
6. compare by request class, not only by overall SLO attainment

## Practical guidance for the current codebase

- Use `priority_fcfs` as the current safer baseline.
- Treat `deadline_fcfs` and `slack_edf` as experimental.
- Do not rely on `slack_edf` without per-request TTFT metadata.
- Expect `slack_edf` to be fragile when the predictor is weak or the workload is
  deeply overloaded.
- When `text` is the product-critical class, check whether aggregate goodput
  gains are coming from sacrificing `text`.
- Keep the current bounded rescue / cooldown logic if you need the latest
  stabilization work, but do not present it as a final tail-latency fix.
- Do not reintroduce the reverted waiting-side medium/long batch-size cap unless
  a new benchmark result shows a clearly different regime.

## Summary

The main limitation exposed by the current implementation is still:

> once tight-SLO short requests are classified as infeasible, the current
> feasible-first `slack_edf` policy has limited ways to recover them.

This is amplified by:

- lightweight remaining-time prediction
- mixed workloads with very different SLOs
- deep overload
- decode-heavy benchmarks

The most useful next steps are:

1. evaluate with `max_new_tokens=1`
2. improve `TTFT_hat`
3. add bounded rescue / anti-starvation logic for short tight-SLO requests
4. evaluate by class, not only aggregate goodput
5. avoid repeating already-reverted batching experiments unless there is new
   evidence that the workload regime has changed
