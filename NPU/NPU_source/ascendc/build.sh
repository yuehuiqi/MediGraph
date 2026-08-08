#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASCEND_HOME="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
SOC_VERSION="${SOC_VERSION:-Ascend910B3}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-1}"

set +u
if [[ -f "${ASCEND_HOME}/bin/setenv.bash" ]]; then
    source "${ASCEND_HOME}/bin/setenv.bash" || true
elif [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh || true
fi
set -u

export ASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME}"

if [[ "${CLEAN_BUILD:-1}" == "1" ]]; then
    rm -rf "${ROOT}/build" "${ROOT}/out"
fi

cmake -S "${ROOT}" -B "${ROOT}/build" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DSOC_VERSION="${SOC_VERSION}" \
    -DRUN_MODE=npu \
    -DASCEND_CANN_PACKAGE_PATH="${ASCEND_CANN_PACKAGE_PATH}"

cmake --build "${ROOT}/build" -- -j"${JOBS}"

echo "BUILD_OK ${ROOT}/build/medical_softmax_run"
