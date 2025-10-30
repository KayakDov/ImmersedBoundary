//
// Created by dov on 10/28/25.
//

#ifndef BICGSTAB_GRIDBOUNDARY_H
#define BICGSTAB_GRIDBOUNDARY_H

#include "deviceArrays/headers/Mat.h"

class GridDim {

public:
    const size_t _rows, _cols, _layers;
    GridDim(size_t rows, size_t cols, size_t layers);
    /**
     * @brief Calculates the total number of unknowns (interior points) in the 3D grid.
     * @return The total grid size: $\text{rows} \times \text{cols} \times \text{layers}$.
     */
    [[nodiscard]] size_t size() const;
};


#endif //BICGSTAB_GRIDBOUNDARY_H
