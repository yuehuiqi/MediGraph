#include <cstddef>
#include <cstdint>
#include <mutex>

#include "acl/acl.h"

extern void medical_softmax_launch(
    uint32_t blockDim, void *stream,
    uint8_t *x, uint8_t *y,
    uint32_t totalRows, uint32_t labels);

namespace {
std::mutex runtimeMutex;
bool initialized = false;
aclrtStream stream = nullptr;
uint8_t *deviceInput = nullptr;
uint8_t *deviceOutput = nullptr;
size_t capacityBytes = 0;

int ensureInitialized() {
    if (initialized) return 0;

    // torch_npu may already have initialized ACL, so repeat-init is ignored.
    (void)aclInit(nullptr);

    aclError status = aclrtSetDevice(0);
    if (status != ACL_SUCCESS) return static_cast<int>(status);

    status = aclrtCreateStream(&stream);
    if (status != ACL_SUCCESS) return static_cast<int>(status);

    initialized = true;
    return 0;
}

int ensureCapacity(size_t bytes) {
    if (bytes <= capacityBytes) return 0;

    if (deviceInput) aclrtFree(deviceInput);
    if (deviceOutput) aclrtFree(deviceOutput);
    deviceInput = nullptr;
    deviceOutput = nullptr;
    capacityBytes = 0;

    aclError status = aclrtMalloc(
        reinterpret_cast<void **>(&deviceInput),
        bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (status != ACL_SUCCESS) return static_cast<int>(status);

    status = aclrtMalloc(
        reinterpret_cast<void **>(&deviceOutput),
        bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (status != ACL_SUCCESS) {
        aclrtFree(deviceInput);
        deviceInput = nullptr;
        return static_cast<int>(status);
    }

    capacityBytes = bytes;
    return 0;
}
}

extern "C" int medical_npu_init() {
    std::lock_guard<std::mutex> lock(runtimeMutex);
    return ensureInitialized();
}

extern "C" int medical_npu_run_f32(
    const float *input,
    float *output,
    uint32_t rows,
    uint32_t labels,
    uint32_t blockDim) {
    if (!input || !output || rows == 0 || labels != 32)
        return -2;

    std::lock_guard<std::mutex> lock(runtimeMutex);

    int status = ensureInitialized();
    if (status != 0) return status;

    size_t bytes =
        static_cast<size_t>(rows) * labels * sizeof(float);

    status = ensureCapacity(bytes);
    if (status != 0) return status;

    aclError aclStatus = aclrtMemcpy(
        deviceInput, bytes, input, bytes,
        ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclStatus != ACL_SUCCESS)
        return static_cast<int>(aclStatus);

    medical_softmax_launch(
        blockDim, stream,
        deviceInput, deviceOutput,
        rows, labels);

    aclStatus = aclrtSynchronizeStream(stream);
    if (aclStatus != ACL_SUCCESS)
        return static_cast<int>(aclStatus);

    aclStatus = aclrtMemcpy(
        output, bytes, deviceOutput, bytes,
        ACL_MEMCPY_DEVICE_TO_HOST);
    return static_cast<int>(aclStatus);
}

extern "C" void medical_npu_finalize() {
    std::lock_guard<std::mutex> lock(runtimeMutex);

    if (deviceInput) aclrtFree(deviceInput);
    if (deviceOutput) aclrtFree(deviceOutput);
    if (stream) aclrtDestroyStream(stream);

    deviceInput = nullptr;
    deviceOutput = nullptr;
    stream = nullptr;
    capacityBytes = 0;
    initialized = false;

    // 不调用 aclFinalize/aclrtResetDevice，避免破坏同进程 torch_npu。
}

extern "C" const char *medical_npu_backend_name() {
    return "ascendc_medical_softmax_c220";
}
