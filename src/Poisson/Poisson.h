//
// Created by dov on 10/28/25.
//

#ifndef BICGSTAB_GRIDBOUNDARY_H
#define BICGSTAB_GRIDBOUNDARY_H

#include "Poisson/CubeBoundary.cu"
#include "deviceArrays/headers/GridDim.h"


template <typename T>
class Poisson {

public:
    const GridDim dim;

protected:
    Vec<T> _b;
    void setB(const CubeBoundary<T>& boundary, cudaStream_t stream);



public:
    Poisson(const CubeBoundary<T>& boundary, Vec<T> &f, cudaStream_t stream);
    /**
     * @brief Calculates the total number of unknowns (interior points) in the 3D grid.
     * @return The total grid size: $\text{rows} \times \text{cols} \times \text{layers}$.
     */
    [[nodiscard]] size_t size() const;
};


#endif //BICGSTAB_GRIDBOUNDARY_H
