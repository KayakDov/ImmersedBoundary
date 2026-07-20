/**
 * @file LaplOperatorType.h
 * @brief Per-axis discretization selector for the separable Laplacian.
 * @ingroup poisson
 *
 * Lives in its own header so that FortranBindings.hpp, BoundaryConfig.cuh and
 * EigenDecompForFortran.h can all use it without including one another
 * (they previously formed an include cycle through the enum).
 */

#ifndef CUDABANDED_LAPLOPERATORTYPE_H
#define CUDABANDED_LAPLOPERATORTYPE_H

namespace eigen {

    /**
     * Discretization used along one axis.
     *  - UniformDeltaNodeCenteredLapl: uniform spacing, unknowns at nodes,
     *    walls ON the first/last node's neighbouring position (the historical
     *    isStaggered == false uniform path).
     *  - UniformDeltaStaggeredLapl: uniform spacing, unknowns at cell centres,
     *    walls half a spacing beyond the first/last unknown (the historical
     *    isStaggered == true uniform path).
     *  - VariableDeltaLapl: variable spacing, pointwise 3-point Laplacian
     *    (exact for quadratics at each node; faces implicitly at midpoints).
     *  - FluxLapl: variable spacing, conservative finite-volume flux form;
     *    unknowns are centres of cells that tile the segment wall-to-wall,
     *    cell widths are reconstructed from the deltas.  Required when the
     *    operator must equal div(grad) discretely (e.g. pressure projection).
     *
     * Values are part of the Fortran ABI (passed as plain integers); do not
     * reorder.
     */
    enum LaplOperatorT {
        UniformDeltaNodeCenteredLapl = 0,
        UniformDeltaStaggeredLapl    = 1,
        VariableDeltaLapl            = 2,
        FluxLapl                     = 3
    };

    inline bool hasVariableDelta(LaplOperatorT segSpacing) {
        return segSpacing == VariableDeltaLapl || segSpacing == FluxLapl;
    }
    inline bool hasVariableDelta(size_t segSpacing) {
        return hasVariableDelta(static_cast<LaplOperatorT>(segSpacing));
    }
}

#endif // CUDABANDED_LAPLOPERATORTYPE_H
