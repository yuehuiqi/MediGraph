#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "acl/acl.h"

extern void medical_softmax_launch(
    uint32_t blockDim, void *stream,
    uint8_t *x, uint8_t *y,
    uint32_t totalRows, uint32_t labels);

#define CHECK(expr)                                                        \
    do {                                                                   \
        aclError _e = (expr);                                              \
        if (_e != ACL_SUCCESS) {                                           \
            std::fprintf(stderr, "ACL error %d at %s:%d\n",                \
                         _e, __FILE__, __LINE__);                           \
            return 1;                                                      \
        }                                                                  \
    } while (0)

using Clock = std::chrono::steady_clock;

static double elapsed_ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static double percentile(std::vector<double> values, double q) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    size_t index = static_cast<size_t>(
        std::ceil(q * values.size()) - 1);
    if (index >= values.size()) index = values.size() - 1;
    return values[index];
}

static double mean(const std::vector<double> &values) {
    double sum = 0.0;
    for (double value : values) sum += value;
    return values.empty() ? 0.0 : sum / values.size();
}

static void cpu_softmax(
    const std::vector<float> &x,
    std::vector<float> &y,
    uint32_t rows, uint32_t labels) {
    for (uint32_t r = 0; r < rows; ++r) {
        const float *src = x.data() + static_cast<size_t>(r) * labels;
        float *dst = y.data() + static_cast<size_t>(r) * labels;

        float maximum = src[0];
        for (uint32_t c = 1; c < labels; ++c)
            maximum = src[c] > maximum ? src[c] : maximum;

        float total = 0.0f;
        for (uint32_t c = 0; c < labels; ++c) {
            dst[c] = std::exp(src[c] - maximum);
            total += dst[c];
        }
        for (uint32_t c = 0; c < labels; ++c)
            dst[c] /= total;
    }
}

int main(int argc, char **argv) {
    uint32_t rows = argc > 1 ? std::atoi(argv[1]) : 200000;
    uint32_t labels = argc > 2 ? std::atoi(argv[2]) : 32;
    uint32_t throughputIters = argc > 3 ? std::atoi(argv[3]) : 200;
    uint32_t blockDim = argc > 4 ? std::atoi(argv[4]) : 40;
    uint32_t latencyIters = argc > 5 ? std::atoi(argv[5]) : 200;
    uint32_t loadSeconds = argc > 6 ? std::atoi(argv[6]) : 0;

    if (rows == 0 || labels == 0 || throughputIters == 0) {
        std::fprintf(stderr, "rows/labels/iters must be positive\n");
        return 2;
    }

    const size_t elements = static_cast<size_t>(rows) * labels;
    const size_t bytes = elements * sizeof(float);

    std::vector<float> input(elements);
    std::vector<float> output(elements);
    std::vector<float> reference(elements);

    std::srand(42);
    for (float &value : input)
        value = static_cast<float>(std::rand()) / RAND_MAX * 6.0f - 3.0f;

    cpu_softmax(input, reference, rows, labels);

    CHECK(aclInit(nullptr));
    CHECK(aclrtSetDevice(0));

    aclrtStream stream = nullptr;
    CHECK(aclrtCreateStream(&stream));

    uint8_t *deviceInput = nullptr;
    uint8_t *deviceOutput = nullptr;

    CHECK(aclrtMalloc(
        reinterpret_cast<void **>(&deviceInput),
        bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(
        reinterpret_cast<void **>(&deviceOutput),
        bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMemcpy(
        deviceInput, bytes, input.data(), bytes,
        ACL_MEMCPY_HOST_TO_DEVICE));

    for (uint32_t i = 0; i < 20; ++i)
        medical_softmax_launch(
            blockDim, stream, deviceInput, deviceOutput, rows, labels);
    CHECK(aclrtSynchronizeStream(stream));

    // Sustained throughput: asynchronous submissions, one final sync.
    auto throughputStart = Clock::now();
    for (uint32_t i = 0; i < throughputIters; ++i)
        medical_softmax_launch(
            blockDim, stream, deviceInput, deviceOutput, rows, labels);
    CHECK(aclrtSynchronizeStream(stream));
    auto throughputEnd = Clock::now();

    double totalSeconds =
        std::chrono::duration<double>(
            throughputEnd - throughputStart).count();
    double throughput =
        static_cast<double>(rows) * throughputIters / totalSeconds;
    double batchMeanMs =
        totalSeconds * 1000.0 / throughputIters;

    // Single-request latency distribution.
    std::vector<double> latencies;
    latencies.reserve(latencyIters);

    for (uint32_t i = 0; i < latencyIters; ++i) {
        auto start = Clock::now();
        medical_softmax_launch(
            blockDim, stream, deviceInput, deviceOutput, rows, labels);
        CHECK(aclrtSynchronizeStream(stream));
        auto end = Clock::now();
        latencies.push_back(elapsed_ms(start, end));
    }

    // End-to-end latency: H2D + kernel + D2H, allocations reused.
    std::vector<double> endToEnd;
    uint32_t e2eIters = std::min<uint32_t>(10, latencyIters);
    endToEnd.reserve(e2eIters);

    for (uint32_t i = 0; i < e2eIters; ++i) {
        auto start = Clock::now();
        CHECK(aclrtMemcpy(
            deviceInput, bytes, input.data(), bytes,
            ACL_MEMCPY_HOST_TO_DEVICE));
        medical_softmax_launch(
            blockDim, stream, deviceInput, deviceOutput, rows, labels);
        CHECK(aclrtSynchronizeStream(stream));
        CHECK(aclrtMemcpy(
            output.data(), bytes, deviceOutput, bytes,
            ACL_MEMCPY_DEVICE_TO_HOST));
        auto end = Clock::now();
        endToEnd.push_back(elapsed_ms(start, end));
    }

    CHECK(aclrtMemcpy(
        output.data(), bytes, deviceOutput, bytes,
        ACL_MEMCPY_DEVICE_TO_HOST));

    double maxAbs = 0.0;
    for (size_t i = 0; i < elements; ++i)
        maxAbs = std::max(
            maxAbs,
            static_cast<double>(
                std::fabs(output[i] - reference[i])));

    double latencyMean = mean(latencies);
    double p50 = percentile(latencies, 0.50);
    double p95 = percentile(latencies, 0.95);
    double p99 = percentile(latencies, 0.99);
    double e2eMean = mean(endToEnd);

    std::printf("=== Ascend C medical_softmax ===\n");
    std::printf(
        "shape=[%u,%u] throughput_iters=%u latency_iters=%u blockDim=%u\n",
        rows, labels, throughputIters, latencyIters, blockDim);
    std::printf("throughput       = %.0f rows/s\n", throughput);
    std::printf("batch mean       = %.3f ms\n", batchMeanMs);
    std::printf("latency mean     = %.3f ms\n", latencyMean);
    std::printf("latency P50/P95/P99 = %.3f / %.3f / %.3f ms\n",
                p50, p95, p99);
    std::printf("end-to-end mean  = %.3f ms\n", e2eMean);
    std::printf("max|diff| vs CPU = %.3e (%s)\n",
                maxAbs, maxAbs <= 1e-5 ? "PASS" : "FAIL");

    const char *jsonFormat =
        "{\"system\":\"npu_ascendc\","
        "\"rows\":%u,\"labels\":%u,\"block_dim\":%u,"
        "\"throughput_rows_per_s\":%.1f,"
        "\"batch_latency_mean_ms\":%.6f,"
        "\"latency_mean_ms\":%.6f,"
        "\"latency_p50_ms\":%.6f,"
        "\"latency_p95_ms\":%.6f,"
        "\"latency_p99_ms\":%.6f,"
        "\"end_to_end_mean_ms\":%.6f,"
        "\"max_abs_diff_vs_cpu\":%.9e}";

    std::printf("RESULT_JSON ");
    std::printf(
        jsonFormat,
        rows, labels, blockDim, throughput, batchMeanMs,
        latencyMean, p50, p95, p99, e2eMean, maxAbs);
    std::printf("\n");
    std::fflush(stdout);

    if (const char *path = std::getenv("ASCENDC_RESULT_PATH")) {
        if (FILE *file = std::fopen(path, "w")) {
            std::fprintf(
                file, jsonFormat,
                rows, labels, blockDim, throughput, batchMeanMs,
                latencyMean, p50, p95, p99, e2eMean, maxAbs);
            std::fprintf(file, "\n");
            std::fclose(file);
        }
    }

    // Sustained load used by the later energy benchmark.
    if (loadSeconds > 0) {
        std::printf("LOAD_READY\n");
        std::fflush(stdout);

        uint64_t launches = 0;
        auto loadStart = Clock::now();
        auto loadEnd = loadStart;

        do {
            for (uint32_t i = 0; i < 128; ++i)
                medical_softmax_launch(
                    blockDim, stream,
                    deviceInput, deviceOutput, rows, labels);
            CHECK(aclrtSynchronizeStream(stream));
            launches += 128;
            loadEnd = Clock::now();
        } while (
            std::chrono::duration<double>(
                loadEnd - loadStart).count() < loadSeconds);

        double loadElapsed =
            std::chrono::duration<double>(
                loadEnd - loadStart).count();

        std::printf(
            "LOAD_JSON {\"launches\":%llu,"
            "\"seconds\":%.3f,\"rows_per_s\":%.1f}\n",
            static_cast<unsigned long long>(launches),
            loadElapsed,
            static_cast<double>(rows) * launches / loadElapsed);
    }

    aclrtFree(deviceInput);
    aclrtFree(deviceOutput);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    return maxAbs <= 1e-5 ? 0 : 3;
}
