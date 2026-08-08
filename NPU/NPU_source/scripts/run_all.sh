#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-full}"

case "${MODE}" in
    quick|full) ;;
    *)
        echo "Usage: $0 [quick|full]"
        exit 2
        ;;
esac

ROWS="${ROWS:-200000}"
LABELS="${LABELS:-32}"
ITERS="${ITERS:-200}"
LATENCY_ITERS="${LATENCY_ITERS:-200}"
REPEATS="${REPEATS:-5}"
BLOCKDIM="${BLOCKDIM:-40}"
TEMPERATURE="${TEMPERATURE:-1.35}"
JOBS="${JOBS:-1}"

PYTORCH_ENV="${PYTORCH_ENV:-/home/ma-user/anaconda3/envs/PyTorch-2.7.1}"

if [[ ! -x "${PYTORCH_ENV}/bin/python" ]]; then
    echo "Python environment missing: ${PYTORCH_ENV}"
    exit 2
fi

export PATH="${PYTORCH_ENV}/bin:${PATH}"
hash -r

RUN_ID="$(
    date +%Y%m%d_%H%M%S
)_${MODE}"

RUN_DIR="${ROOT}/artifacts/runs/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"

mkdir -p \
  "${LOG_DIR}" \
  "${RUN_DIR}/repeats/pure" \
  "${RUN_DIR}/repeats/e2e" \
  "${RUN_DIR}/profiler/pure" \
  "${RUN_DIR}/profiler/fused"

if [[ -L "${ROOT}/results" ]]; then
    rm -- "${ROOT}/results"
elif [[ -e "${ROOT}/results" ]]; then
    echo "Refusing to replace non-symlink results directory"
    exit 2
fi

ln -s "artifacts/runs/${RUN_ID}" \
  "${ROOT}/results"

ln -sfn "runs/${RUN_ID}" \
  "${ROOT}/artifacts/latest"

run_logged() {
    local name="$1"
    shift

    echo
    echo "===== ${name} ====="
    echo "command: $*"

    set +e
    "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
    local status="${PIPESTATUS[0]}"
    set -e

    if [[ "${status}" -ne 0 ]]; then
        echo "FAILED ${name}: ${status}"
        return "${status}"
    fi

    echo "PASSED ${name}"
}

set +u
ASCEND_HOME="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"

if [[ -f "${ASCEND_HOME}/bin/setenv.bash" ]]; then
    source "${ASCEND_HOME}/bin/setenv.bash" || true
elif [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh || true
fi
set -u

cat >"${RUN_DIR}/run_config.json" <<JSON
{
  "run_id": "${RUN_ID}",
  "mode": "${MODE}",
  "rows": ${ROWS},
  "labels": ${LABELS},
  "iterations": ${ITERS},
  "latency_iterations": ${LATENCY_ITERS},
  "repeats": ${REPEATS},
  "block_dim": ${BLOCKDIM},
  "temperature": ${TEMPERATURE}
}
JSON

npu-smi info \
  >"${RUN_DIR}/npu_smi_before.txt"

python3 - <<'PY' \
  >"${RUN_DIR}/environment.json" \
  2>"${LOG_DIR}/environment.stderr.log"
import json
import os
import platform
import subprocess
import sys

import numpy
import torch

try:
    import torch_npu
    torch_npu_version = getattr(
        torch_npu, "__version__", "unknown"
    )
except Exception as error:
    torch_npu_version = repr(error)

def output(command):
    try:
        return subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as error:
        return repr(error)

report = {
    "platform": platform.platform(),
    "machine": platform.machine(),
    "python": sys.version,
    "numpy": numpy.__version__,
    "torch": torch.__version__,
    "torch_npu": torch_npu_version,
    "npu_available":
        hasattr(torch, "npu")
        and torch.npu.is_available(),
    "cmake": output(["cmake", "--version"]).splitlines()[0],
    "gcc": output(["gcc", "--version"]).splitlines()[0],
    "npu_smi_version":
        output(["npu-smi", "info"]).splitlines()[0],
    "environment": {
        name: os.environ.get(name, "")
        for name in (
            "ASCEND_HOME_PATH",
            "ASCEND_OPP_PATH",
            "PYTHONPATH",
            "LD_LIBRARY_PATH",
        )
    },
}

print(json.dumps(report, indent=2))
PY

run_logged build \
  env CLEAN_BUILD=1 JOBS="${JOBS}" \
  "${ROOT}/ascendc/build.sh"

run_logged integration_pure \
  python "${ROOT}/integration/npu_softmax_operator.py"

run_logged integration_fused \
  python "${ROOT}/integration/npu_fused_operator.py"

if [[ "${MODE}" == "full" ]]; then
    run_logged bench_softmax_energy \
      python "${ROOT}/benchmark/bench_softmax.py" \
      --energy
fi

PURE_REPEATS="${REPEATS}"
if [[ "${MODE}" == "quick" ]]; then
    PURE_REPEATS=1
fi

for run in $(seq 1 "${PURE_REPEATS}"); do
    run_logged "pure_repeat_${run}" \
      python "${ROOT}/benchmark/compare_all.py" \
      --rows "${ROWS}" \
      --labels "${LABELS}" \
      --iters "${ITERS}" \
      --blockdim "${BLOCKDIM}"

    cp "${RUN_DIR}/compare_all.json" \
      "${RUN_DIR}/repeats/pure/run_${run}.json"
done

run_logged fused_standalone \
  "${ROOT}/ascendc/run_fused.sh" \
  "${ROWS}" "${LABELS}" "${ITERS}" \
  "${BLOCKDIM}" "${LATENCY_ITERS}" \
  "${TEMPERATURE}" 0

FUSED_REPEATS="${REPEATS}"
if [[ "${MODE}" == "quick" ]]; then
    FUSED_REPEATS=1
fi

run_logged fused_repeated \
  python "${ROOT}/benchmark/compare_fused.py" \
  --rows "${ROWS}" \
  --labels "${LABELS}" \
  --iters "${ITERS}" \
  --latency-iters "${LATENCY_ITERS}" \
  --repeats "${FUSED_REPEATS}" \
  --blockdim "${BLOCKDIM}" \
  --temperature "${TEMPERATURE}"

if [[ "${MODE}" == "full" ]]; then
    for run in $(seq 1 "${REPEATS}"); do
        run_logged "e2e_repeat_${run}" \
          python "${ROOT}/benchmark/end_to_end_compare.py"

        cp "${RUN_DIR}/end_to_end_compare.json" \
          "${RUN_DIR}/repeats/e2e/run_${run}.json"
    done

    run_logged pure_energy \
      python "${ROOT}/benchmark/energy_compare.py"

    run_logged fused_energy \
      python "${ROOT}/benchmark/energy_compare_fused.py" \
      --rows "${ROWS}" \
      --idle-seconds 10 \
      --load-seconds 20 \
      --temperature "${TEMPERATURE}"

    run_logged profiler_pure \
      msprof op \
      --output="${RUN_DIR}/profiler/pure" \
      "${ROOT}/ascendc/build/medical_softmax_run" \
      "${ROWS}" "${LABELS}" 1 \
      "${BLOCKDIM}" 1 0

    run_logged profiler_fused \
      msprof op \
      --output="${RUN_DIR}/profiler/fused" \
      "${ROOT}/ascendc/build/medical_fused_softmax_run" \
      "${ROWS}" "${LABELS}" 1 \
      "${BLOCKDIM}" 1 "${TEMPERATURE}" 0
fi

run_logged summarize \
  python "${ROOT}/scripts/summarize_run.py" \
  "${RUN_DIR}" "${MODE}"

(
    cd "${ROOT}"

    {
        find \
          ascendc benchmark integration scripts \
          -type f \
          ! -path 'ascendc/build/*' \
          ! -path '*/__pycache__/*' \
          ! -name '*.pyc' \
          -print0

        for file in \
          run_all.sh \
          README.md \
          REPRODUCE.md \
          .gitignore
        do
            [[ -f "${file}" ]] &&
              printf '%s\0' "${file}"
        done
    } |
      sort -z |
      xargs -0 sha256sum
) >"${RUN_DIR}/source.sha256"

npu-smi info \
  >"${RUN_DIR}/npu_smi_after.txt"

PACKAGE_LOG="$(mktemp)"

"${ROOT}/scripts/package_run.sh" "${RUN_DIR}" \
  2>&1 | tee "${PACKAGE_LOG}"

mv "${PACKAGE_LOG}" \
  "${LOG_DIR}/package.log"

echo
echo "FULL_RUN_OK"
echo "RUN_ID=${RUN_ID}"
echo "RUN_DIR=${RUN_DIR}"
echo "REPORT=${RUN_DIR}/REPORT.md"
echo "SUMMARY=${RUN_DIR}/summary.json"
echo "LATEST=${ROOT}/artifacts/latest"
