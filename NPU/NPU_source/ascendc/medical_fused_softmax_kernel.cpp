#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t FUSED_BUFFER_NUM = 2;
constexpr uint32_t FUSED_TILE_ROWS = 248;
constexpr uint32_t FP32_PER_DATABLOCK = 8;

template <typename T>
class MedicalFusedSoftmaxKernel {
public:
    __aicore__ inline MedicalFusedSoftmaxKernel() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR labelMask,
        GM_ADDR y,
        uint32_t totalRows,
        uint32_t labels,
        float invTemperature) {
        this->labels = labels;
        this->invTemperature = invTemperature;

        uint32_t coreNum = GetBlockNum();
        uint32_t rowsPerCore =
            (totalRows + coreNum - 1) / coreNum;
        uint32_t startRow =
            GetBlockIdx() * rowsPerCore;

        rowsThisCore =
            startRow < totalRows
                ? ((startRow + rowsPerCore <= totalRows)
                       ? rowsPerCore
                       : totalRows - startRow)
                : 0;

        uint32_t offset = startRow * labels;

        xGm.SetGlobalBuffer(
            (__gm__ T *)x + offset,
            rowsThisCore * labels);

        yGm.SetGlobalBuffer(
            (__gm__ T *)y + offset,
            rowsThisCore * labels);

        maskGm.SetGlobalBuffer(
            (__gm__ T *)labelMask,
            labels);

        uint32_t tileBytes =
            FUSED_TILE_ROWS * labels * sizeof(T);

        pipe.InitBuffer(
            inQueue, FUSED_BUFFER_NUM, tileBytes);
        pipe.InitBuffer(
            outQueue, FUSED_BUFFER_NUM, tileBytes);
        pipe.InitBuffer(
            maskQueue, 1, labels * sizeof(T));
        pipe.InitBuffer(
            reduceBuf,
            FUSED_TILE_ROWS * sizeof(T));
        pipe.InitBuffer(
            broadcastBuf,
            FUSED_TILE_ROWS *
                FP32_PER_DATABLOCK * sizeof(T));
    }

    __aicore__ inline void Process() {
        LocalTensor<T> maskLocal =
            maskQueue.AllocTensor<T>();

        DataCopy(maskLocal, maskGm, labels);
        maskQueue.EnQue(maskLocal);

        LocalTensor<T> residentMask =
            maskQueue.DeQue<T>();

        uint32_t rowOffset = 0;

        while (rowOffset < rowsThisCore) {
            uint32_t remaining =
                rowsThisCore - rowOffset;
            uint32_t rowCount =
                remaining < FUSED_TILE_ROWS
                    ? remaining
                    : FUSED_TILE_ROWS;

            CopyIn(rowOffset, rowCount);
            Compute(rowCount, residentMask);
            CopyOut(rowOffset, rowCount);

            rowOffset += rowCount;
        }

        maskQueue.FreeTensor(residentMask);
    }

private:
    __aicore__ inline void CopyIn(
        uint32_t rowOffset,
        uint32_t rowCount) {
        LocalTensor<T> xLocal =
            inQueue.AllocTensor<T>();

        DataCopy(
            xLocal,
            xGm[rowOffset * labels],
            rowCount * labels);

        inQueue.EnQue(xLocal);
    }

    __aicore__ inline void Compute(
        uint32_t rowCount,
        const LocalTensor<T> &maskLocal) {
        LocalTensor<T> xLocal =
            inQueue.DeQue<T>();
        LocalTensor<T> yLocal =
            outQueue.AllocTensor<T>();
        LocalTensor<T> reduced =
            reduceBuf.Get<T>();
        LocalTensor<T> broadcast =
            broadcastBuf.Get<T>();

        uint8_t rowRepeats =
            static_cast<uint8_t>(rowCount);
        uint8_t brcbRepeats =
            static_cast<uint8_t>(
                (rowCount + 7) / 8);
        uint8_t rowBlocks =
            static_cast<uint8_t>(
                labels * sizeof(T) / 32);

        // Temperature calibration:
        // calibrated = logits / temperature.
        Muls<T>(
            yLocal,
            xLocal,
            static_cast<T>(invTemperature),
            rowCount * labels);

        // Reuse one [32] mask for every row.
        // dst=y, src0=mask, src1=y.
        BinaryRepeatParams maskBroadcast = {
            1, 1, 1,
            rowBlocks, 0, rowBlocks
        };

        Add<T>(
            yLocal,
            maskLocal,
            yLocal,
            static_cast<uint64_t>(labels),
            rowRepeats,
            maskBroadcast);

        WholeReduceMax<T>(
            reduced,
            yLocal,
            static_cast<int32_t>(labels),
            rowRepeats,
            1, 1, rowBlocks,
            ReduceOrder::ORDER_ONLY_VALUE);

        Brcb<T>(
            broadcast,
            reduced,
            brcbRepeats,
            {1, 8});

        BinaryRepeatParams rowBroadcast = {
            1, 1, 0,
            rowBlocks, rowBlocks, 1
        };

        // Reuse input UB as temporary storage.
        Sub<T>(
            xLocal,
            yLocal,
            broadcast,
            static_cast<uint64_t>(labels),
            rowRepeats,
            rowBroadcast);

        Exp<T>(
            xLocal,
            xLocal,
            rowCount * labels);

        WholeReduceSum<T>(
            reduced,
            xLocal,
            static_cast<int32_t>(labels),
            rowRepeats,
            1, 1, rowBlocks);

        Brcb<T>(
            broadcast,
            reduced,
            brcbRepeats,
            {1, 8});

        Div<T>(
            yLocal,
            xLocal,
            broadcast,
            static_cast<uint64_t>(labels),
            rowRepeats,
            rowBroadcast);

        outQueue.EnQue<T>(yLocal);
        inQueue.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(
        uint32_t rowOffset,
        uint32_t rowCount) {
        LocalTensor<T> yLocal =
            outQueue.DeQue<T>();

        DataCopy(
            yGm[rowOffset * labels],
            yLocal,
            rowCount * labels);

        outQueue.FreeTensor(yLocal);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, FUSED_BUFFER_NUM>
        inQueue;
    TQue<QuePosition::VECOUT, FUSED_BUFFER_NUM>
        outQueue;
    TQue<QuePosition::VECIN, 1> maskQueue;
    TBuf<QuePosition::VECCALC> reduceBuf;
    TBuf<QuePosition::VECCALC> broadcastBuf;

    GlobalTensor<T> xGm;
    GlobalTensor<T> maskGm;
    GlobalTensor<T> yGm;

    uint32_t labels = 0;
    uint32_t rowsThisCore = 0;
    float invTemperature = 1.0f;
};

extern "C" __global__ __aicore__
void medical_fused_softmax(
    GM_ADDR x,
    GM_ADDR labelMask,
    GM_ADDR y,
    uint32_t totalRows,
    uint32_t labels,
    float invTemperature) {
    MedicalFusedSoftmaxKernel<float> op;
    op.Init(
        x, labelMask, y,
        totalRows, labels,
        invTemperature);
    op.Process();
}

#ifndef __CCE_KT_TEST__
void medical_fused_softmax_launch(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *labelMask,
    uint8_t *y,
    uint32_t totalRows,
    uint32_t labels,
    float invTemperature) {
    medical_fused_softmax
        <<<blockDim, nullptr, stream>>>(
            x,
            labelMask,
            y,
            totalRows,
            labels,
            invTemperature);
}
#endif
