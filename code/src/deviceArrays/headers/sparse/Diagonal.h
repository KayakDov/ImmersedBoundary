/**
 * @file Diagonal.h
 * @brief Defines the diagonal sparse-matrix specialization.
 * @ingroup sparse_matrices
 *
 * @details
 * Sparse-matrix classes build on the device array layer and expose storage-format-specific operations while preserving explicit CUDA handle and stream control.
 */


#ifndef CUDABANDED_DIAG_H
#define CUDABANDED_DIAG_H
#include "deviceArrays/headers/sparse/BandedMat.h"

template <typename T>
class Diagonal : public BandedMat<T>{
public:
    inline static const AdjacencyInd primary{0, 0};

    static const Singleton<int32_t> &getSharedIndices(Handle &hand);

    /**
     * @brief Allocates an uninitialized diagonal matrix directly on device memory.
     * @param denseSqMatDim Matrix dimension count (rows/columns).
     * @param hand The active CUDA stream/context handle.
     */
    Diagonal(size_t denseSqMatDim, Handle &hand);

    /**
     * @brief Wraps an existing allocated matrix buffer as a diagonal matrix.
     * @param windowInto Input Matrix buffer with exactly 1 column.
     * @param hand The active CUDA stream/context handle.
     */
    Diagonal(const Mat<T> &windowInto, Handle &hand);


    /**
     * @brief Wraps an existing contiguous array as the primary diagonal.
     * @param windowInto Contiguous device array used as the diagonal storage.
     * @param hand Active CUDA stream/context handle.
     */
    Diagonal(const SimpleArray<T> &windowInto, Handle &hand);

};



#endif //CUDABANDED_DIAG_H
