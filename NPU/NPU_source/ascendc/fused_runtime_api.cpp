#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mutex>

#include "acl/acl.h"

extern void medical_fused_softmax_launch(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *labelMask,
    uint8_t *y,
    uint32_t totalRows,
    uint32_t labels,
    float invTemperature);

namespace {
std::mutex mutex;
bool initialized = false;
aclrtStream stream = nullptr;

uint8_t *deviceInput = nullptr;
uint8_t *deviceMask = nullptr;
uint8_t *deviceOutput = nullptr;
size_t capacityBytes = 0;

int ensureInitialized() {
    if (initialized) return 0;

    (void)aclInit(nullptr);

    aclError status = aclrtSetDevice(0);
    if (status != ACL_SUCCESS)
        return static_cast<int>(status);

    status = aclrtCreateStream(&stream);
    if (status != ACL_SUCCESS)
        return static_cast<int>(status);

    initialized = true;
    return 0;
}

void releaseBuffers() {
    if (deviceInput) aclrtFree(deviceInput);
    if (deviceMask) aclrtFree(deviceMask);
    if (deviceOutput) aclrtFree(deviceOutput);

    deviceInput = nullptr;
    deviceMask = nullptr;
    deviceOutput = nullptr;
    capacityBytes = 0;
}

int ensureCapacity(size_t bytes) {
    if (bytes <= capacityBytes &&
        deviceInput && deviceMask && deviceOutput)
        return 0;

    releaseBuffers();

    aclError status = aclrtMalloc(
        reinterpret_cast<void **>(&deviceInput),
        bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (status != ACL_SUCCESS)
        return static_cast<int>(status);

    status = aclrtMalloc(
        reinterpret_cast<void **>(&deviceOutput),
        bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (status != ACL_SUCCESS) {
        releaseBuffers();
        return static_cast<int>(status);
    }

    status = aclrtMalloc(
        reinterpret_cast<void **>(&deviceMask),
        32 * sizeof(float),
        ACL_MEM_MALLOC_HUGE_FIRST);
    if (status != ACL_SUCCESS) {
        releaseBuffers();
        return static_cast<int>(status);
    }

    capacityBytes = bytes;
    return 0;
}

bool invalidArguments(
    uint32_t rows,
    uint32_t labels,
    uint32_t blockDim,
    float temperature) {
    return rows == 0 ||
           labels != 32 ||
           blockDim == 0 ||
           !std::isfinite(temperature) ||
           temperature <= 0.0f;
}
}

extern "C" int medical_npu_fused_init() {
    std::lock_guard<std::mutex> lock(mutex);
    return ensureInitialized();
}

extern "C" int medical_npu_fused_run_f32(
    const float *input,
    const float *additiveMask,
    float *output,
    uint32_t rows,
    uint32_t labels,
    uint32_t blockDim,
    float temperature) {
    if (!input || !additiveMask || !output ||
        invalidArguments(rows, labels, blockDim, temperature))
        return -2;

    std::lock_guard<std::mutex> lock(mutex);

    int status = ensureInitialized();
    if (status != 0) return status;

    size_t bytes =
        static_cast<size_t>(rows) *
        labels * sizeof(float);

    status = ensureCapacity(bytes);
    if (status != 0) return status;

    aclError aclStatus = aclrtMemcpy(
        deviceInput, bytes,
        input, bytes,
        ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclStatus != ACL_SUCCESS)
        return static_cast<int>(aclStatus);

    aclStatus = aclrtMemcpy(
        deviceMask, labels * sizeof(float),
        additiveMask, labels * sizeof(float),
        ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclStatus != ACL_SUCCESS)
        return static_cast<int>(aclStatus);

    medical_fused_softmax_launch(
        blockDim, stream,
        deviceInput, deviceMask, deviceOutput,
        rows, labels, 1.0f / temperature);

    aclStatus = aclrtSynchronizeStream(stream);
    if (aclStatus != ACL_SUCCESS)
        return static_cast<int>(aclStatus);

    aclStatus = aclrtMemcpy(
        output, bytes,
        deviceOutput, bytes,
        ACL_MEMCPY_DEVICE_TO_HOST);

    return static_cast<int>(aclStatus);
}

extern "C" int
medical_npu_fused_run_device_async_f32(
    uintptr_t inputAddress,
    uintptr_t maskAddress,
    uintptr_t outputAddress,
    uint32_t rows,
    uint32_t labels,
    uint32_t blockDim,
    float temperature) {
    if (!inputAddress || !maskAddress || !outputAddress ||
        invalidArguments(rows, labels, blockDim, temperature))
        return -2;

    std::lock_guard<std::mutex> lock(mutex);

    int status = ensureInitialized();
    if (status != 0) return status;

    medical_fused_softmax_launch(
        blockDim, stream,
        reinterpret_cast<uint8_t *>(inputAddress),
        reinterpret_cast<uint8_t *>(maskAddress),
        reinterpret_cast<uint8_t *>(outputAddress),
        rows, labels, 1.0f / temperature);

    return 0;
}

extern "C" int medical_npu_fused_synchronize() {
    std::lock_guard<std::mutex> lock(mutex);

    int status = ensureInitialized();
    if (status != 0) return status;

    return static_cast<int>(
        aclrtSynchronizeStream(stream));
}

extern "C" void medical_npu_fused_finalize() {
    std::lock_guard<std::mutex> lock(mutex);

    if (stream)
        (void)aclrtSynchronizeStream(stream);

    releaseBuffers();

    if (stream)
        aclrtDestroyStream(stream);

    stream = nullptr;
    initialized = false;
}

extern "C" const char *
medical_npu_fused_backend_name() {
    return "ascendc_medical_calibrated_masked_softmax";
}
