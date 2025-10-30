
#include "deviceArrays/headers/vec.h"
#include "GridDim.h"
#include "deviceArrays/headers/squareMat.h"

template <typename T>
class FastDiagonalizationMethod:  GridDim{
private:

    std::array<SquareMat<T>, 3> I;
    std::array<SquareMat<T>, 3> L;

    void setLI(size_t i, cudaStream_t stream) {//TODO: try parallelizing this data
        I[i].setToIdentity(stream);
        L[i].fill(static_cast<T>(0), stream);
        L[i].diag(0).fill(static_cast<T>(-2), stream);
        L[i].diag(1).fill(static_cast<T>(1), stream);
        L[i].diag(-1).fill(static_cast<T>(1), stream);
    }

    /**
     * The kronecker product of a, b, and c
     * @param a
     * @param b
     * @param c
     * @param temp should be a.rows * b.rows x a.cols * b.cols
     * @param result should be a.rows * b.rows * c.rows x a.cols * b.cols * c.cols
     * @param stream
     */
    void kronecker3(const Mat<T>& a, const Mat<T>& b, const Mat<T>& c, Mat<T>& temp, Mat<T>& result, cudaStream_t stream) {
        temp.fill(0, stream);
        a.multKronecker(b, temp, stream);
        temp.multKronecker(c, result, stream);
    }

public:
    FastDiagonalizationMethod(size_t rows, size_t cols, size_t layers) :
        GridDim(rows, cols, layers),
        I({SquareMat<T>::create(cols), SquareMat<T>::create(rows),  SquareMat<T>::create(layers)}),
        L({SquareMat<T>::create(cols), SquareMat<T>::create(rows), SquareMat<T>::create(layers)}){

        Handle hand;

        for (size_t i = 0; i < 3; ++i) setLI(i, hand.stream);

        SquareMat<T> maybeA = SquareMat<T>::create(rows * cols * layers);
        maybeA.fill(static_cast<T>(0), hand.stream);

        auto rowsXColsTemp = SquareMat<T>::create(cols * rows);
        kronecker3(L[0], I[1], I[2], rowsXColsTemp, maybeA, hand.stream);
        kronecker3(I[0], L[1], I[2], rowsXColsTemp, maybeA, hand.stream);
        kronecker3(I[0], I[1], L[2], rowsXColsTemp, maybeA, hand.stream);

        maybeA.get(std::cout, true, false, hand.stream);

    }
};


/**
 * @brief Main entry point to demonstrate the FastDiagonalizationMethod for a 2x2x2 grid.
 */
int main() {

    // Define the 2x2x2 grid dimensions
    constexpr size_t dim = 2;
    constexpr  size_t rows = dim;
    constexpr  size_t cols = dim;
    constexpr  size_t layers = dim;

    std::cout << "Starting Fast Diagonalization Method setup for a "
              << rows << "x" << cols << "x" << layers << " grid (Total size: "
              << rows * cols * layers << ")\n\n";

    FastDiagonalizationMethod<double> fdm(rows, cols, layers);

    return 0;
}