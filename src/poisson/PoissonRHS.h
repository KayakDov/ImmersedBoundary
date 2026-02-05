//
// Created by dov on 10/28/25.
//

#ifndef BICGSTAB_GRIDBOUNDARY_H
#define BICGSTAB_GRIDBOUNDARY_H

#include "poisson/CubeBoundary.h"
#include "../deviceArrays/headers/Support/GridDim.hpp"


/**
 *TODO: this should really just be a function that sets the b from the RHS of the Poisson equation.
 *
 * Note, this is for the 3d problem.  All dimensions are assumed to be greater than 1.
 * @tparam T
 */
template <typename T>
class PoissonRHS {

public:
    const GridDim dim;

protected:
    Vec<T> _b;

public:
    /**
     *
     * @param boundary  The boundary conditions.
     * @param f The right hand side operator.  This will be overwritten.
     * @param stream
     */
    PoissonRHS(const CubeBoundary<T>& boundary, Vec<T> &f, cudaStream_t stream);

};


#endif //BICGSTAB_GRIDBOUNDARY_H
