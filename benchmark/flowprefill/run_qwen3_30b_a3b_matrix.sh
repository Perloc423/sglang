#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  benchmark/flowprefill/run_qwen3_30b_a3b_matrix.sh --phase PHASE [options]

This script wraps workload generation and replay commands for:
  /mnt/cephfs/wangyong/models/hf/Qwen3-30B-A3B-Instruct-2507

Phases:
  generate_workloads        Build QwenTrace-derived workload JSONL files.
  rate_sweep                Replay one system across request-rate scales.
  slo_sweep                 Replay one system across SLO scales.
  full_output               Replay one system with original output lengths.

Required:
  --phase PHASE             One of the phases above.

Optional:
  --model-path PATH         Default: /mnt/cephfs/wangyong/models/hf/Qwen3-30B-A3B-Instruct-2507
  --trace-path PATH         Required for generate_workloads.
  --system-tag TAG          baseline / flowprefill_priority_fcfs / flowprefill_deadline_fcfs / flowprefill_slack_edf
                            Required for replay phases.
  --host HOST               Default: 127.0.0.1
  --port PORT               Default: 30000
  --window-start-seconds N  Default: 300
  --time-window-seconds N   Default: 120
  --max-requests N          Optional cap after time-window selection.
  --output-dir DIR          Default: benchmark/flowprefill/results/qwen3_30b_a3b
  --workload-dir DIR        Default: benchmark/flowprefill/workloads/qwen3_30b_a3b
  --model-preset PRESET     Default: qwen2.5-14b
  --rates "LIST"            Space-separated rates. Default: "0.5 1.0 2.0"
  --slo-scales "LIST"       Space-separated scales. Default: "1.0 2.0 4.0 8.0"
  --fixed-slo-scale FLOAT   For rate_sweep. Default: 4.0
  --fixed-rate FLOAT        For slo_sweep/full_output. Default: 1.0
  --print-window-summary    Forward to replay client.
  --ignore-eos              Forward to replay client.
  --filename-suffix TAG     Extra suffix added to replay output files.

Examples:
  benchmark/flowprefill/run_qwen3_30b_a3b_matrix.sh \
    --phase generate_workloads \
    --trace-path /data/qwen_traceA_blksz_16.jsonl

  benchmark/flowprefill/run_qwen3_30b_a3b_matrix.sh \
    --phase rate_sweep \
    --system-tag flowprefill_slack_edf
EOF
}

PHASE=""
MODEL_PATH="/mnt/cephfs/wangyong/models/hf/Qwen3-30B-A3B-Instruct-2507"
TRACE_PATH=""
SYSTEM_TAG=""
HOST="127.0.0.1"
PORT="30000"
WINDOW_START_SECONDS="300"
TIME_WINDOW_SECONDS="120"
MAX_REQUESTS=""
OUTPUT_DIR="benchmark/flowprefill/results/qwen3_30b_a3b"
WORKLOAD_DIR="benchmark/flowprefill/workloads/qwen3_30b_a3b"
MODEL_PRESET="qwen2.5-14b"
RATES="0.5 1.0 2.0"
SLO_SCALES="1.0 2.0 4.0 8.0"
FIXED_SLO_SCALE="4.0"
FIXED_RATE="1.0"
PRINT_WINDOW_SUMMARY=0
IGNORE_EOS=0
FILENAME_SUFFIX=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase)
      PHASE="$2"
      shift 2
      ;;
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --trace-path)
      TRACE_PATH="$2"
      shift 2
      ;;
    --system-tag)
      SYSTEM_TAG="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --window-start-seconds)
      WINDOW_START_SECONDS="$2"
      shift 2
      ;;
    --time-window-seconds)
      TIME_WINDOW_SECONDS="$2"
      shift 2
      ;;
    --max-requests)
      MAX_REQUESTS="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --workload-dir)
      WORKLOAD_DIR="$2"
      shift 2
      ;;
    --model-preset)
      MODEL_PRESET="$2"
      shift 2
      ;;
    --rates)
      RATES="$2"
      shift 2
      ;;
    --slo-scales)
      SLO_SCALES="$2"
      shift 2
      ;;
    --fixed-slo-scale)
      FIXED_SLO_SCALE="$2"
      shift 2
      ;;
    --fixed-rate)
      FIXED_RATE="$2"
      shift 2
      ;;
    --print-window-summary)
      PRINT_WINDOW_SUMMARY=1
      shift
      ;;
    --ignore-eos)
      IGNORE_EOS=1
      shift
      ;;
    --filename-suffix)
      FILENAME_SUFFIX="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${PHASE}" ]]; then
  echo "--phase is required." >&2
  usage >&2
  exit 1
fi

mkdir -p "${WORKLOAD_DIR}" "${OUTPUT_DIR}"

run_replay() {
  local experiment_tag="$1"
  local workload_file="$2"
  local slo_scale="$3"
  local rate="$4"
  local decode_override="$5"

  local cmd=(
    bash benchmark/flowprefill/run_trace_replay.sh
    --host "${HOST}"
    --port "${PORT}"
    --model "${MODEL_PATH}"
    --workload-file "${workload_file}"
    --output-dir "${OUTPUT_DIR}"
    --experiment-tag "${experiment_tag}"
    --system-tag "${SYSTEM_TAG}"
    --slo-scale "${slo_scale}"
    --window-start-seconds "${WINDOW_START_SECONDS}"
    --request-rate-scale "${rate}"
  )

  if [[ -n "${TIME_WINDOW_SECONDS}" ]]; then
    cmd+=(--time-window-seconds "${TIME_WINDOW_SECONDS}")
  fi

  if [[ -n "${MAX_REQUESTS}" ]]; then
    cmd+=(--max-requests "${MAX_REQUESTS}")
  fi

  if [[ -n "${decode_override}" ]]; then
    cmd+=(--override-max-new-tokens "${decode_override}")
  fi

  if [[ "${PRINT_WINDOW_SUMMARY}" -eq 1 ]]; then
    cmd+=(--print-window-summary)
  fi

  if [[ "${IGNORE_EOS}" -eq 1 ]]; then
    cmd+=(--ignore-eos)
  fi

  if [[ -n "${FILENAME_SUFFIX}" ]]; then
    cmd+=(--filename-suffix "${FILENAME_SUFFIX}")
  fi

  printf 'Running:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
}

case "${PHASE}" in
  generate_workloads)
    if [[ -z "${TRACE_PATH}" ]]; then
      echo "--trace-path is required for generate_workloads." >&2
      exit 1
    fi

    for slo_scale in ${SLO_SCALES}; do
      output_file="${WORKLOAD_DIR}/qwentrace_qwen3_30b_a3b_slo${slo_scale}.jsonl"
      python benchmark/flowprefill/build_qwentrace_workload.py \
        --input "${TRACE_PATH}" \
        --output "${output_file}" \
        --model-preset "${MODEL_PRESET}" \
        --slo-scale "${slo_scale}"
    done
    ;;

  rate_sweep)
    if [[ -z "${SYSTEM_TAG}" ]]; then
      echo "--system-tag is required for rate_sweep." >&2
      exit 1
    fi

    workload_file="${WORKLOAD_DIR}/qwentrace_qwen3_30b_a3b_slo${FIXED_SLO_SCALE}.jsonl"
    for rate in ${RATES}; do
      run_replay "rate_sweep" "${workload_file}" "${FIXED_SLO_SCALE}" "${rate}" "1"
    done
    ;;

  slo_sweep)
    if [[ -z "${SYSTEM_TAG}" ]]; then
      echo "--system-tag is required for slo_sweep." >&2
      exit 1
    fi

    for slo_scale in ${SLO_SCALES}; do
      workload_file="${WORKLOAD_DIR}/qwentrace_qwen3_30b_a3b_slo${slo_scale}.jsonl"
      run_replay "slo_sweep" "${workload_file}" "${slo_scale}" "${FIXED_RATE}" "1"
    done
    ;;

  full_output)
    if [[ -z "${SYSTEM_TAG}" ]]; then
      echo "--system-tag is required for full_output." >&2
      exit 1
    fi

    workload_file="${WORKLOAD_DIR}/qwentrace_qwen3_30b_a3b_slo${FIXED_SLO_SCALE}.jsonl"
    for rate in ${RATES}; do
      run_replay "full_output" "${workload_file}" "${FIXED_SLO_SCALE}" "${rate}" ""
    done
    ;;

  *)
    echo "Unsupported phase: ${PHASE}" >&2
    usage >&2
    exit 1
    ;;
esac
