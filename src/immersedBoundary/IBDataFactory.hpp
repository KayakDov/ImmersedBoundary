// #include <vector>
// #include <cmath>
// #include <numeric>
// #include <algorithm>
//
// #include "GridDim.hpp"
// #include "SparseCSC.cuh"
// #include "math/Real3d.h"
//
// /**
//  * @brief Factory for Immersed Boundary data using physical grid spacing.
//  * * @tparam Real Floating point precision (float or double).
//  * @tparam Ind Integer type for indexing (int, size_t).
//  */
// template <typename Real, typename Ind>
// class IBDataFactory {
// public:
//     const GridDim dim;
//
//     SparseMatrixBuilder<Real, Ind> sparseBuilder;
//     std::vector<Real> f;
//     std::vector<Real> p;
//
//     double delta(size_t flatInd, const Real3d& center) {
//         size_t row = flatInd % dim.rows;
//         size_t col = (flatInd / dim.rows) % dim.cols;
//         size_t layer = flatInd % dim.layerSize;
//         double r = (center - Real3d(col, row, layer)).normInf();
//         if (r > 1.5 ) return 0;
//         if (r > 0.5) return 1.0/6 * (5 - 3 * r - std::sqrt(-3 *(1 - r)*(1 - r) + 1));
//         return  1.0/3 * (1 + std::sqrt(-3*r*r + 1));
//     }
//
//
//     void buildBRow(size_t fRow, const Real3d& loc) {
//
//     }
//
//     /**
//      * @brief Constructor initializing data based on physical dimensions.
//      * * @param gridDim Eulerian grid dimensions.
//      * @param gridDelta Physical spacing (dx, dy, dz).
//      */
//     IBDataFactory(const GridDim& gridDim)
//         : dim(gridDim) {
//
//         double r = gridDim.minDim()/4.0;
//         size_t numSurfacePoints = static_cast<size_t>(2 * M_PI * r * (gridDim.numDims() == 3 ? 2 * r: 1));
//         p.assign(dim.volume(), 0.0);
//         f.assign(2 * numSurfacePoints, 0.0);
//         std::fill(f.begin() + numSurfacePoints, f.end(), 1.0);
//
//     }
//
// };