#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASCEND_HOME="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"

N="${1:-200000}"
C="${2:-32}"
ITERS="${3:-200}"
BLOCKDIM="${4:-40}"
LATENCY_ITERS="${5:-200}"
TEMPERATURE="${6:-1.35}"
LOAD_SECONDS="${7:-0}"

set +u
if [[ -f "${ASCEND_HOME}/bin/setenv.bash" ]]; then
    source "${ASCEND_HOME}/bin/setenv.bash" || true
elif [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh || true
fi
set -u

if [[ ! -x "${ROOT}/build/medical_fused_softmax_run" ]] ||
   [[ "${REBUILD:-0}" == "1" ]]; then
    "${ROOT}/build.sh"
fi

mkdir -p "${ROOT}/../results"
export ASCENDC_FUSED_RESULT_PATH="${ROOT}/../results/fused_ascendc.json"

cd "${ROOT}/build"

exec ./medical_fused_softmax_run \
    "${N}" "${C}" "${ITERS}" "${BLOCKDIM}" \
    "${LATENCY_ITERS}" "${TEMPERATURE}" "${LOAD_SECONDS}"
