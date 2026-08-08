#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

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

#define CHECK_ACL(expr)                                      \
    do {                                                     \
        aclError status_ = (expr);                           \
        if (status_ != ACL_SUCCESS) {                        \
            std::fprintf(                                    \
                stderr,                                      \
                "ACL error %d at %s:%d\n",                   \
                static_cast<int>(status_),                   \
                __FILE__, __LINE__);                         \
            return 2;                                        \
        }                                                    \
    } while (0)

static void cpuReference(
    const std::vector<float> &x,
    const std::vector<float> &mask,
    std::vector<float> &y,
    uint32_t rows,
    uint32_t labels,
    float temperature) {
    for (uint32_t row = 0; row < rows; ++row) {
        float maximum = -INFINITY;

        for (uint32_t label = 0;
             label < labels; ++label) {
            float value =
                x[row * labels + label] /
                    temperature +
                mask[label];
            maximum = std::max(maximum, value);
        }

        float sum = 0.0f;

        for (uint32_t label = 0;
             label < labels; ++label) {
            float value =
                x[row * labels + label] /
                    temperature +
                mask[label];

            float exponent =
                std::exp(value - maximum);

            y[row * labels + label] = exponent;
            sum += exponent;
        }

        for (uint32_t label = 0;
             label < labels; ++label) {
            y[row * labels + label] /= sum;
        }
    }
}

static double percentile(
    std::vector<double> values,
    double quantile) {
    std::sort(values.begin(), values.end());

    size_t index = static_cast<size_t>(
        std::ceil(quantile * values.size()));

    if (index == 0) index = 1;
    index -= 1;

    return values[
        std::min(index, values.size() - 1)];
}

static double mean(
    const std::vector<double> &values) {
    return std::accumulate(
               values.begin(),
               values.end(),
               0.0) /
           values.size();
}

int main(int argc, char **argv) {
    uint32_t rows =
        argc > 1 ? std::atoi(argv[1]) : 200000;
    uint32_t labels =
        argc > 2 ? std::atoi(argv[2]) : 32;
    uint32_t throughputIters =
        argc > 3 ? std::atoi(argv[3]) : 200;
    uint32_t blockDim =
        argc > 4 ? std::atoi(argv[4]) : 40;
    uint32_t latencyIters =
        argc > 5 ? std::atoi(argv[5]) : 200;
    float temperature =
        argc > 6
            ? std::atof(argv[6])
            : 1.35f;
    double loadSeconds =
        argc > 7
            ? std::atof(argv[7])
            : 0.0;

    if (rows == 0 ||
        labels != 32 ||
        throughputIters == 0 ||
        latencyIters == 0 ||
        !std::isfinite(temperature) ||
        temperature <= 0.0f) {
        std::fprintf(
            stderr,
            "requires rows>0, labels=32, "
            "iters>0, temperature>0\n");
        return 1;
    }

    size_t elements =
        static_cast<size_t>(rows) * labels;
    size_t bytes =
        elements * sizeof(float);
    size_t maskBytes =
        labels * sizeof(float);

    std::vector<float> input(elements);
    std::vector<float> output(elements);
    std::vector<float> reference(elements);
    std::vector<float> labelMask(labels, 0.0f);

    std::mt19937 generator(42);
    std::uniform_real_distribution<float>
        distribution(-3.0f, 3.0f);

    for (float &value : input)
        value = distribution(generator);

    // Reproducible ontology constraint:
    // classes 0,7,14,21,28 are invalid.
    for (uint32_t label = 0;
         label < labels; ++label) {
        if (label % 7 == 0)
            labelMask[label] = -10000.0f;
    }

    cpuReference(
        input,
        labelMask,
        reference,
        rows,
        labels,
        temperature);

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));

    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *deviceInput = nullptr;
    uint8_t *deviceMask = nullptr;
    uint8_t *deviceOutput = nullptr;

    CHECK_ACL(aclrtMalloc(
        reinterpret_cast<void **>(&deviceInput),
        bytes,
        ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMalloc(
        reinterpret_cast<void **>(&deviceMask),
        maskBytes,
        ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMalloc(
        reinterpret_cast<void **>(&deviceOutput),
        bytes,
        ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMemcpy(
        deviceInput, bytes,
        input.data(), bytes,
        ACL_MEMCPY_HOST_TO_DEVICE));

    CHECK_ACL(aclrtMemcpy(
        deviceMask, maskBytes,
        labelMask.data(), maskBytes,
        ACL_MEMCPY_HOST_TO_DEVICE));

    float invTemperature =
        1.0f / temperature;

    for (int index = 0; index < 20; ++index) {
        medical_fused_softmax_launch(
            blockDim,
            stream,
            deviceInput,
            deviceMask,
            deviceOutput,
            rows,
            labels,
            invTemperature);
    }

    CHECK_ACL(aclrtSynchronizeStream(stream));

    auto throughputStart =
        std::chrono::steady_clock::now();

    for (uint32_t index = 0;
         index < throughputIters;
         ++index) {
        medical_fused_softmax_launch(
            blockDim,
            stream,
            deviceInput,
            deviceMask,
            deviceOutput,
            rows,
            labels,
            invTemperature);
    }

    CHECK_ACL(aclrtSynchronizeStream(stream));

    auto throughputEnd =
        std::chrono::steady_clock::now();

    double throughputSeconds =
        std::chrono::duration<double>(
            throughputEnd - throughputStart)
            .count();

    double throughput =
        static_cast<double>(rows) *
        throughputIters /
        throughputSeconds;

    double batchMeanMs =
        throughputSeconds /
        throughputIters *
        1000.0;

    std::vector<double> latencies;
    latencies.reserve(latencyIters);

    for (uint32_t index = 0;
         index < latencyIters;
         ++index) {
        auto start =
            std::chrono::steady_clock::now();

        medical_fused_softmax_launch(
            blockDim,
            stream,
            deviceInput,
            deviceMask,
            deviceOutput,
            rows,
            labels,
            invTemperature);

        CHECK_ACL(
            aclrtSynchronizeStream(stream));

        auto end =
            std::chrono::steady_clock::now();

        latencies.push_back(
            std::chrono::duration<double,
                                  std::milli>(
                end - start)
                .count());
    }

    // Pageable H2D + kernel + D2H.
    // The label mask is treated as resident configuration.
    std::vector<double> endToEndTimes;
    endToEndTimes.reserve(20);

    for (int index = 0; index < 20; ++index) {
        auto start =
            std::chrono::steady_clock::now();

        CHECK_ACL(aclrtMemcpy(
            deviceInput, bytes,
            input.data(), bytes,
            ACL_MEMCPY_HOST_TO_DEVICE));

        medical_fused_softmax_launch(
            blockDim,
            stream,
            deviceInput,
            deviceMask,
            deviceOutput,
            rows,
            labels,
            invTemperature);

        CHECK_ACL(
            aclrtSynchronizeStream(stream));

        CHECK_ACL(aclrtMemcpy(
            output.data(), bytes,
            deviceOutput, bytes,
            ACL_MEMCPY_DEVICE_TO_HOST));

        auto end =
            std::chrono::steady_clock::now();

        endToEndTimes.push_back(
            std::chrono::duration<double,
                                  std::milli>(
                end - start)
                .count());
    }

    double maxAbs = 0.0;
    double rowSumError = 0.0;
    double maskedProbabilityMax = 0.0;

    for (uint32_t row = 0;
         row < rows; ++row) {
        double rowSum = 0.0;

        for (uint32_t label = 0;
             label < labels; ++label) {
            size_t offset =
                static_cast<size_t>(row) *
                    labels +
                label;

            maxAbs = std::max(
                maxAbs,
                static_cast<double>(
                    std::fabs(
                        output[offset] -
                        reference[offset])));

            rowSum += output[offset];

            if (labelMask[label] < -1000.0f) {
                maskedProbabilityMax =
                    std::max(
                        maskedProbabilityMax,
                        static_cast<double>(
                            std::fabs(
                                output[offset])));
            }
        }

        rowSumError = std::max(
            rowSumError,
            std::fabs(rowSum - 1.0));
    }

    double latencyMean = mean(latencies);
    double p50 = percentile(latencies, 0.50);
    double p95 = percentile(latencies, 0.95);
    double p99 = percentile(latencies, 0.99);
    double endToEndMean = mean(endToEndTimes);

    std::printf(
        "=== Ascend C fused medical softmax ===\n");
    std::printf(
        "operation=temperature+label_mask+softmax\n");
    std::printf(
        "shape=[%u,32] temperature=%.3f "
        "blockDim=%u\n",
        rows, temperature, blockDim);
    std::printf(
        "throughput       = %.0f rows/s\n",
        throughput);
    std::printf(
        "batch mean       = %.6f ms\n",
        batchMeanMs);
    std::printf(
        "latency mean     = %.6f ms\n",
        latencyMean);
    std::printf(
        "latency P50/P95/P99 = "
        "%.6f / %.6f / %.6f ms\n",
        p50, p95, p99);
    std::printf(
        "end-to-end mean  = %.6f ms\n",
        endToEndMean);
    std::printf(
        "max|diff| vs CPU = %.9e\n",
        maxAbs);
    std::printf(
        "row sum error    = %.9e\n",
        rowSumError);
    std::printf(
        "masked prob max  = %.9e\n",
        maskedProbabilityMax);

    std::printf(
        "FUSED_RESULT_JSON "
        "{\"system\":\"npu_ascendc_fused\","
        "\"operation\":\"temperature+label_mask+softmax\","
        "\"rows\":%u,"
        "\"labels\":32,"
        "\"temperature\":%.6f,"
        "\"block_dim\":%u,"
        "\"throughput_rows_per_s\":%.1f,"
        "\"batch_latency_mean_ms\":%.9f,"
        "\"latency_mean_ms\":%.9f,"
        "\"latency_p50_ms\":%.9f,"
        "\"latency_p95_ms\":%.9f,"
        "\"latency_p99_ms\":%.9f,"
        "\"end_to_end_mean_ms\":%.9f,"
        "\"max_abs_diff_vs_cpu\":%.9e,"
        "\"row_sum_error\":%.9e,"
        "\"masked_probability_max\":%.9e}\n",
        rows,
        temperature,
        blockDim,
        throughput,
        batchMeanMs,
        latencyMean,
        p50,
        p95,
        p99,
        endToEndMean,
        maxAbs,
        rowSumError,
        maskedProbabilityMax);

    const char *resultPath =
        std::getenv(
            "ASCENDC_FUSED_RESULT_PATH");

    if (!resultPath)
        resultPath =
            "../../results/fused_ascendc.json";

    FILE *resultFile =
        std::fopen(resultPath, "w");

    if (resultFile) {
        std::fprintf(
            resultFile,
            "{\"system\":\"npu_ascendc_fused\","
            "\"operation\":\"temperature+label_mask+softmax\","
            "\"rows\":%u,"
            "\"labels\":32,"
            "\"temperature\":%.6f,"
            "\"block_dim\":%u,"
            "\"throughput_rows_per_s\":%.1f,"
            "\"batch_latency_mean_ms\":%.9f,"
            "\"latency_mean_ms\":%.9f,"
            "\"latency_p50_ms\":%.9f,"
            "\"latency_p95_ms\":%.9f,"
            "\"latency_p99_ms\":%.9f,"
            "\"end_to_end_mean_ms\":%.9f,"
            "\"max_abs_diff_vs_cpu\":%.9e,"
            "\"row_sum_error\":%.9e,"
            "\"masked_probability_max\":%.9e}\n",
            rows,
            temperature,
            blockDim,
            throughput,
            batchMeanMs,
            latencyMean,
            p50,
            p95,
            p99,
            endToEndMean,
            maxAbs,
            rowSumError,
            maskedProbabilityMax);

        std::fclose(resultFile);
    }

    if (loadSeconds > 0.0) {
        std::printf("LOAD_READY\n");
        std::fflush(stdout);

        uint64_t launches = 0;
        auto loadStart =
            std::chrono::steady_clock::now();
        double elapsed = 0.0;

        do {
            for (int batch = 0;
                 batch < 200;
                 ++batch) {
                medical_fused_softmax_launch(
                    blockDim,
                    stream,
                    deviceInput,
                    deviceMask,
                    deviceOutput,
                    rows,
                    labels,
                    invTemperature);
                ++launches;
            }

            CHECK_ACL(
                aclrtSynchronizeStream(stream));

            elapsed =
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() -
                    loadStart)
                    .count();
        } while (elapsed < loadSeconds);

        std::printf(
            "LOAD_JSON "
            "{\"system\":\"npu_ascendc_fused\","
            "\"launches\":%llu,"
            "\"seconds\":%.6f,"
            "\"rows_per_s\":%.1f}\n",
            static_cast<unsigned long long>(
                launches),
            elapsed,
            static_cast<double>(rows) *
                launches / elapsed);
    }

    bool passed =
        maxAbs <= 1e-5 &&
        rowSumError <= 1e-5 &&
        maskedProbabilityMax <= 1e-6;

    aclrtFree(deviceInput);
    aclrtFree(deviceMask);
    aclrtFree(deviceOutput);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    return passed ? 0 : 3;
}
