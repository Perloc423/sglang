#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  benchmark/flowprefill/run_trace_replay.sh --model MODEL --workload-file FILE [options]

Required:
  --model MODEL                     Served model id or model path passed to /generate.
  --workload-file FILE              Input workload JSONL.

Optional:
  --host HOST                       Server host. Default: 127.0.0.1
  --port PORT                       Server port. Default: 30000
  --skip-flush-cache                Do not call /flush_cache before replay.
  --output-dir DIR                  Result directory. Default: benchmark/flowprefill/results
  --system-tag TAG                  System label, e.g. baseline / flowprefill_slack_edf.
                                    Default: unnamed
  --experiment-tag TAG              Extra label for grouping runs. Default: replay
  --window-start-seconds FLOAT      Default: 0
  --time-window-seconds FLOAT       Optional contiguous replay window length.
  --request-rate-scale FLOAT        Default: 1.0
  --slo-scale FLOAT                 Optional label only, used in output filename.
  --override-max-new-tokens INT     Optional; set to 1 for decode-controlled TTFT runs.
  --max-requests INT                Optional cap after window selection.
  --ignore-eos                      Forward to replay client.
  --print-window-summary            Forward to replay client.
  --filename-suffix TAG             Optional extra suffix appended to output filename.

Examples:
  benchmark/flowprefill/run_trace_replay.sh \
    --model Qwen/Qwen2.5-14B-Instruct \
    --workload-file benchmark/flowprefill/workloads/qwen2p5_14b/qwentrace_slo4.0.jsonl \
    --system-tag flowprefill_slack_edf \
    --experiment-tag rate_sweep \
    --slo-scale 4.0 \
    --window-start-seconds 300 \
    --time-window-seconds 120 \
    --request-rate-scale 2.0 \
    --override-max-new-tokens 1
EOF
}

HOST="127.0.0.1"
PORT="30000"
SKIP_FLUSH_CACHE=0
OUTPUT_DIR="benchmark/flowprefill/results"
SYSTEM_TAG="unnamed"
EXPERIMENT_TAG="replay"
WINDOW_START_SECONDS="0"
TIME_WINDOW_SECONDS=""
REQUEST_RATE_SCALE="1.0"
SLO_SCALE=""
OVERRIDE_MAX_NEW_TOKENS=""
MAX_REQUESTS=""
IGNORE_EOS=0
PRINT_WINDOW_SUMMARY=0
FILENAME_SUFFIX=""
MODEL=""
WORKLOAD_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --workload-file)
      WORKLOAD_FILE="$2"
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
    --skip-flush-cache)
      SKIP_FLUSH_CACHE=1
      shift
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --system-tag)
      SYSTEM_TAG="$2"
      shift 2
      ;;
    --experiment-tag)
      EXPERIMENT_TAG="$2"
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
    --request-rate-scale)
      REQUEST_RATE_SCALE="$2"
      shift 2
      ;;
    --slo-scale)
      SLO_SCALE="$2"
      shift 2
      ;;
    --override-max-new-tokens)
      OVERRIDE_MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --max-requests)
      MAX_REQUESTS="$2"
      shift 2
      ;;
    --ignore-eos)
      IGNORE_EOS=1
      shift
      ;;
    --print-window-summary)
      PRINT_WINDOW_SUMMARY=1
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

if [[ -z "${MODEL}" || -z "${WORKLOAD_FILE}" ]]; then
  echo "--model and --workload-file are required." >&2
  usage >&2
  exit 1
fi

sanitize_tag() {
  echo "$1" | tr '/ :' '___'
}

mkdir -p "${OUTPUT_DIR}"

SYSTEM_TAG_SAFE="$(sanitize_tag "${SYSTEM_TAG}")"
EXPERIMENT_TAG_SAFE="$(sanitize_tag "${EXPERIMENT_TAG}")"
RATE_TAG="rate${REQUEST_RATE_SCALE}"
WINDOW_TAG="ws${WINDOW_START_SECONDS}"

if [[ -n "${TIME_WINDOW_SECONDS}" ]]; then
  WINDOW_TAG="${WINDOW_TAG}_tw${TIME_WINDOW_SECONDS}"
fi

if [[ -n "${SLO_SCALE}" ]]; then
  SLO_TAG="slo${SLO_SCALE}"
else
  SLO_TAG="sloNA"
fi

if [[ -n "${OVERRIDE_MAX_NEW_TOKENS}" ]]; then
  DECODE_TAG="decode${OVERRIDE_MAX_NEW_TOKENS}"
else
  DECODE_TAG="decodefull"
fi

OUTPUT_BASENAME="${EXPERIMENT_TAG_SAFE}_${SYSTEM_TAG_SAFE}_${SLO_TAG}_${RATE_TAG}_${WINDOW_TAG}_${DECODE_TAG}"
if [[ -n "${FILENAME_SUFFIX}" ]]; then
  OUTPUT_BASENAME="${OUTPUT_BASENAME}_$(sanitize_tag "${FILENAME_SUFFIX}")"
fi
OUTPUT_FILE="${OUTPUT_DIR}/${OUTPUT_BASENAME}.json"

CMD=(
  python benchmark/flowprefill/bench_flowprefill_trace_replay.py
  --host "${HOST}"
  --port "${PORT}"
  --model "${MODEL}"
  --workload-file "${WORKLOAD_FILE}"
  --output-file "${OUTPUT_FILE}"
  --window-start-seconds "${WINDOW_START_SECONDS}"
  --request-rate-scale "${REQUEST_RATE_SCALE}"
)

if [[ -n "${TIME_WINDOW_SECONDS}" ]]; then
  CMD+=(--time-window-seconds "${TIME_WINDOW_SECONDS}")
fi

if [[ -n "${OVERRIDE_MAX_NEW_TOKENS}" ]]; then
  CMD+=(--override-max-new-tokens "${OVERRIDE_MAX_NEW_TOKENS}")
fi

if [[ -n "${MAX_REQUESTS}" ]]; then
  CMD+=(--max-requests "${MAX_REQUESTS}")
fi

if [[ "${IGNORE_EOS}" -eq 1 ]]; then
  CMD+=(--ignore-eos)
fi

if [[ "${PRINT_WINDOW_SUMMARY}" -eq 1 ]]; then
  CMD+=(--print-window-summary)
fi

echo "Running replay client:"
printf '  %q' "${CMD[@]}"
printf '\n'
echo "Output file: ${OUTPUT_FILE}"

if [[ "${SKIP_FLUSH_CACHE}" -ne 1 ]]; then
  FLUSH_URL="http://${HOST}:${PORT}/flush_cache"
  echo "Flushing cache: ${FLUSH_URL}"
  curl --fail --silent --show-error -X POST "${FLUSH_URL}" >/dev/null
fi

"${CMD[@]}"
