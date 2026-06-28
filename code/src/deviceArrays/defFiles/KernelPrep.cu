#include "../headers/Support/KernelPrep.cuh"

/**
 * @brief Adjusts a block dimension to ensure the corresponding grid dimension fits within
 *        the CUDA device limits.
 *
 * This function increases the size of the primary block dimension (`blockSide`) as needed
 * to reduce the number of blocks (grid size) in the corresponding dimension so that it
 * does not exceed `maxGridDim`. The adjustment respects the CUDA limit on the maximum
 * number of threads per block.
 *
 * @param numThreadsNeeded        The total number of elements in this dimension.
 * @param threadsPerBlock   Reference to the primary block dimension to adjust.
 *                    Will be modified if necessary to satisfy hardware limits.
 * @param threadsPerBlockDim2 The size of the block in the first other dimension (Y or X).
 * @param threadsPerBlockDim3 The size of the block in the second other dimension (Z or Y).
 * @param maxNumberOfBlocks  Maximum allowable grid size in this dimension (from device properties,
 *                    e.g., KernelPrep::maxGridX/Y/Z()).
 *
 * @throw std::runtime_error If the required grid cannot fit within device limits
 *        even with the minimum possible block size.
 */
inline void adjustBlockDimToFitGridLimit(
    const size_t numThreadsNeeded,
    size_t &threadsPerBlock,
    const size_t threadsPerBlockDim2,
    const size_t threadsPerBlockDim3,
    const size_t maxNumberOfBlocks // pass maxGridX/Y/Z here
)
{
    size_t numBlocks = KernelPrep::ceiling_division(numThreadsNeeded, threadsPerBlock);

    const size_t maxThreads = KernelPrep::maxThreadsPerBlock();
    const size_t otherThreadsPerBlock = threadsPerBlockDim2 * threadsPerBlockDim3;

    while (numBlocks > maxNumberOfBlocks) {

        threadsPerBlock *= 2;

        if (threadsPerBlock * otherThreadsPerBlock > maxThreads)
             throw std::runtime_error("adjustBlockDimToFitGridLimit: cannot fit grid within device limits");

        numBlocks = KernelPrep::ceiling_division(numThreadsNeeded, threadsPerBlock);
    }
}

KernelPrep::KernelPrep(size_t count) {
    if (count == 0) {
        threadsPerBlock = dim3(0, 0, 0);
        numBlocks = dim3(0, 0, 0);
        return;
    }

    size_t blockX = DEFAULT_BLOCK_SIZE_1D;

    adjustBlockDimToFitGridLimit(count, blockX, 1, 1, maxNumBlocksX());

    threadsPerBlock = dim3(blockX, 1, 1);
    numBlocks  = dim3(ceiling_division(count, blockX), 1, 1);
}

KernelPrep::KernelPrep(size_t width, size_t height, bool transpose) {
    if (width == 0 || height == 0) {
        threadsPerBlock = dim3(0, 0, 0);
        numBlocks  = dim3(0, 0, 0);
        return;
    }

    size_t blockX = DEFAULT_BLOCK_SIZE_2D_SIDE;
    size_t blockY = DEFAULT_BLOCK_SIZE_2D_SIDE;

    adjustBlockDimToFitGridLimit(width, blockX, blockY, 1, transpose? maxNumBlocksY():maxNumBlocksX());
    adjustBlockDimToFitGridLimit(height, blockY, blockX, 1, transpose? maxNumBlocksX():maxNumBlocksY());

    if (transpose) threadsPerBlock = dim3(blockY, blockX, 1);
    else threadsPerBlock = dim3(blockX, blockY, 1);

    size_t numBlocksX = ceiling_division(width, blockX), numBlocksY = ceiling_division(height, blockY);

    if (transpose) numBlocks  = dim3(numBlocksY, numBlocksX, 1);
    else numBlocks  = dim3(numBlocksX, numBlocksY, 1);
}

KernelPrep::KernelPrep(const size_t width, const size_t height, const size_t depth) {
    if (width == 0 || height == 0 || depth == 0) {
        threadsPerBlock = dim3(0, 0, 0);
        numBlocks  = dim3(0, 0, 0);
        return;
    }

    size_t blockX = DEFAULT_BLOCK_SIZE_3D_SIDE;
    size_t blockY = DEFAULT_BLOCK_SIZE_3D_SIDE;
    size_t blockZ = DEFAULT_BLOCK_SIZE_3D_SIDE;

    adjustBlockDimToFitGridLimit(width, blockX, blockY, blockZ, maxNumBlocksX());
    adjustBlockDimToFitGridLimit(height, blockY, blockX, blockZ, maxNumBlocksY());
    adjustBlockDimToFitGridLimit(depth, blockZ, blockX, blockY, maxNumBlocksZ());

    threadsPerBlock = dim3(blockX, blockY, blockZ);
    numBlocks  = dim3(ceiling_division(width, blockX),
                    ceiling_division(height, blockY),
                    ceiling_division(depth, blockZ));
}