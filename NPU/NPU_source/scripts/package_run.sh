#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${1:-$(readlink -f "${ROOT}/artifacts/latest")}"
RUN_DIR="$(readlink -f "${RUN_DIR}")"

case "${RUN_DIR}" in
    "${ROOT}"/artifacts/runs/*) ;;
    *)
        echo "Unsafe run directory: ${RUN_DIR}"
        exit 2
        ;;
esac

if [[ ! -d "${RUN_DIR}" ]]; then
    echo "Run directory does not exist: ${RUN_DIR}"
    exit 2
fi

RUN_ID="$(basename "${RUN_DIR}")"
EXPORT_ROOT="${EXPORT_ROOT:-/home/ma-user/work/NPU_exports}"
EXPORT_DIR="${EXPORT_ROOT}/${RUN_ID}"

mkdir -p "${EXPORT_DIR}"

SOURCE_ARCHIVE="${EXPORT_DIR}/NPU_source_${RUN_ID}.tar.gz"
RESULT_ARCHIVE="${EXPORT_DIR}/NPU_results_${RUN_ID}.tar.gz"

tar \
  --exclude='./ascendc/build' \
  --exclude='./artifacts' \
  --exclude='./results' \
  --exclude='./.git' \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.bak' \
  -czf "${SOURCE_ARCHIVE}" \
  -C "${ROOT}" .

tar \
  -czf "${RESULT_ARCHIVE}" \
  -C "${RUN_DIR}" .

(
    cd "${EXPORT_DIR}"
    sha256sum \
      "$(basename "${SOURCE_ARCHIVE}")" \
      "$(basename "${RESULT_ARCHIVE}")" \
      >SHA256SUMS
)

echo "EXPORT_DIR=${EXPORT_DIR}"
echo "SOURCE=${SOURCE_ARCHIVE}"
echo "RESULTS=${RESULT_ARCHIVE}"
echo "CHECKSUMS=${EXPORT_DIR}/SHA256SUMS"
