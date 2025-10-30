
#include "GridDim.h"

template <typename T>
class FastDiagonalizationMethod:  GridDim{
public:
    FastDiagonalizationMethod(size_t rows, size_t cols, size_t layers)
        : GridDim(rows, cols, layers) {
    }
};