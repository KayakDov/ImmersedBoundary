
#ifndef BICGSTAB_KERNELSUPPORT_CUH
#define BICGSTAB_KERNELSUPPORT_CUH


class DenseInd {
public:
    const int32_t d, row, col;
    __device__ DenseInd(size_t bandedRow, size_t bandedCol, const int32_t* indices):
        d(indices[bandedRow]),
        row(static_cast<int32_t>(d > 0 ? bandedCol : bandedCol - d)),
        col(static_cast<int32_t>(d > 0 ? bandedCol + d : bandedCol))
    {}
    __device__ bool outOfBounds(size_t max) const {
        return row < 0 || row >= max || col < 0 || col >= max;
    }
    __device__ size_t flat(const size_t denseLd) const {
        return col * denseLd + row;
    }
};


#endif //BICGSTAB_KERNELSUPPORT_CUH