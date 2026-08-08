#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_ROWS = 248;
constexpr uint32_t FP32_PER_BLOCK = 8;

template <typename T>
class MedicalSoftmaxKernel {
public:
    __aicore__ inline MedicalSoftmaxKernel() {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, uint32_t totalRows, uint32_t labels) {
        this->labels = labels;

        uint32_t coreNum = GetBlockNum();
        uint32_t rowsPerCore = (totalRows + coreNum - 1) / coreNum;
        uint32_t startRow = GetBlockIdx() * rowsPerCore;

        rowsThisCore =
            startRow < totalRows
                ? ((startRow + rowsPerCore <= totalRows)
                       ? rowsPerCore
                       : totalRows - startRow)
                : 0;

        uint32_t offset = startRow * labels;
        xGm.SetGlobalBuffer((__gm__ T *)x + offset,
                            rowsThisCore * labels);
        yGm.SetGlobalBuffer((__gm__ T *)y + offset,
                            rowsThisCore * labels);

        uint32_t tileBytes = TILE_ROWS * labels * sizeof(T);

        pipe.InitBuffer(inQueue, BUFFER_NUM, tileBytes);
        pipe.InitBuffer(outQueue, BUFFER_NUM, tileBytes);

        // One reduced scalar per row.
        pipe.InitBuffer(reduceBuf, TILE_ROWS * sizeof(T));

        // Brcb expands every scalar to one 32-byte block.
        pipe.InitBuffer(
            broadcastBuf,
            TILE_ROWS * FP32_PER_BLOCK * sizeof(T));
    }

    __aicore__ inline void Process() {
        uint32_t rowOffset = 0;

        while (rowOffset < rowsThisCore) {
            uint32_t remaining = rowsThisCore - rowOffset;
            uint32_t rowCount =
                remaining < TILE_ROWS ? remaining : TILE_ROWS;

            CopyIn(rowOffset, rowCount);
            Compute(rowCount);
            CopyOut(rowOffset, rowCount);

            rowOffset += rowCount;
        }
    }

private:
    __aicore__ inline void CopyIn(
        uint32_t rowOffset, uint32_t rowCount) {
        LocalTensor<T> xLocal = inQueue.AllocTensor<T>();

        DataCopy(xLocal,
                 xGm[rowOffset * labels],
                 rowCount * labels);

        inQueue.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t rowCount) {
        LocalTensor<T> xLocal = inQueue.DeQue<T>();
        LocalTensor<T> yLocal = outQueue.AllocTensor<T>();
        LocalTensor<T> reduced = reduceBuf.Get<T>();
        LocalTensor<T> broadcast = broadcastBuf.Get<T>();

        uint8_t rowRepeats = static_cast<uint8_t>(rowCount);
        uint8_t brcbRepeats =
            static_cast<uint8_t>((rowCount + 7) / 8);

        uint8_t rowBlocks = static_cast<uint8_t>(
            labels * sizeof(T) / 32);

        // One maximum for every row.
        WholeReduceMax<T>(
            reduced, xLocal,
            static_cast<int32_t>(labels),
            rowRepeats,
            1, 1, rowBlocks,
            ReduceOrder::ORDER_ONLY_VALUE);

        // Convert packed maxima into one broadcast block per row.
        Brcb<T>(broadcast, reduced, brcbRepeats, {1, 8});

        BinaryRepeatParams rowBroadcast = {
            1, 1, 0,
            rowBlocks, rowBlocks, 1
        };

        // y = x - row_max
        Sub<T>(
            yLocal, xLocal, broadcast,
            static_cast<uint64_t>(labels),
            rowRepeats, rowBroadcast);

        Exp<T>(yLocal, yLocal, rowCount * labels);

        // One sum for every row.
        WholeReduceSum<T>(
            reduced, yLocal,
            static_cast<int32_t>(labels),
            rowRepeats,
            1, 1, rowBlocks);

        Brcb<T>(broadcast, reduced, brcbRepeats, {1, 8});

        // y = exp(x-max) / row_sum
        Div<T>(
            yLocal, yLocal, broadcast,
            static_cast<uint64_t>(labels),
            rowRepeats, rowBroadcast);

        outQueue.EnQue<T>(yLocal);
        inQueue.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(
        uint32_t rowOffset, uint32_t rowCount) {
        LocalTensor<T> yLocal = outQueue.DeQue<T>();

        DataCopy(yGm[rowOffset * labels],
                 yLocal,
                 rowCount * labels);

        outQueue.FreeTensor(yLocal);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    TBuf<QuePosition::VECCALC> reduceBuf;
    TBuf<QuePosition::VECCALC> broadcastBuf;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    uint32_t labels = 0;
    uint32_t rowsThisCore = 0;
};

extern "C" __global__ __aicore__ void medical_softmax(
    GM_ADDR x, GM_ADDR y,
    uint32_t totalRows, uint32_t labels) {
    MedicalSoftmaxKernel<float> op;
    op.Init(x, y, totalRows, labels);
    op.Process();
}

#ifndef __CCE_KT_TEST__
void medical_softmax_launch(
    uint32_t blockDim, void *stream,
    uint8_t *x, uint8_t *y,
    uint32_t totalRows, uint32_t labels) {
    medical_softmax<<<blockDim, nullptr, stream>>>(
        x, y, totalRows, labels);
}
#endif
