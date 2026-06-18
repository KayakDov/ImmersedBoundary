
#ifndef CUDABANDED_DIAG_H
#define CUDABANDED_DIAG_H
#include "deviceArrays/headers/sparse/BandedMat.h"

template <typename T>
class Diagonal : public BandedMat<T>{
public:
    inline static const AdjacencyInd primary{0, 0};

    static const Singleton<int32_t> &getSharedIndices(Handle &hand);

    /**
     * @brief Allocates an uninitialized tridiagonal matrix directly on device memory.
     * @param denseSqMatDim Matrix dimension count (rows/columns).
     * @param hand The active CUDA stream/context handle.
     */
    Diagonal(size_t denseSqMatDim, Handle &hand);

    /**
     * @brief Wraps an existing allocated matrix asset as a Tridiagonal system.
     * @param windowInto Input Matrix buffer with exactly 1 column.
     * @param hand The active CUDA stream/context handle.
     */
    Diagonal(const Mat<T> &windowInto, Handle &hand);


    Diagonal(const SimpleArray<T> &windowInto, Handle &hand);

};



#endif //CUDABANDED_DIAG_H
