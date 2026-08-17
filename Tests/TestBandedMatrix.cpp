#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>

// Framework Headers
#include "deviceArrays/headers/sparse/BandedMat.h"
#include "deviceArrays/headers/Mat.h"
#include "deviceArrays/headers/Vec.h"
#include "deviceArrays/headers/Singleton.h"
#include "deviceArrays/headers/handle.h"
#include "deviceArrays/headers/Support/Streamable.h"
#include "deviceArrays/headers/SquareMat.h"

template <typename T>
class BandedMatWrappersTest : public ::testing::Test {
protected:
    Handle handle;

    const size_t N = 3;

    // Framework expects padding at the trailing edge of each column.
    // Diag -1: [3, 6, x]
    // Diag  0: [2, 4, 7]
    // Diag  1: [1, 5, x]
    std::vector<int32_t> h_diags = {-1, 0, 1};
    std::vector<T> h_banded_data = {
        3.0, 6.0, 0.0,
        2.0, 4.0, 7.0,
        1.0, 5.0, 0.0
    };

    // Safely constructs and initializes a BandedMat using framework tools
    BandedMat<T> createTestBandedMat() {
        // 1. Create and populate the indices vector
        auto diagsVec = Vec<int32_t>::create(h_diags.size(), this->handle);
        diagsVec.set(h_diags.data(), this->handle);

        // 2. Create the BandedMat framework object
        auto A = BandedMat<T>::create(N, diagsVec, handle);

        // 3. Upload the dense diagonal data to the device
        A.set(h_banded_data.data(), this->handle);

        // std::cout << "A = \n" << GpuOut<T>(dense, handle) << std::endl;

        return A;
    }
};

// Test both single and double precision
typedef ::testing::Types<float, double> Implementations;
TYPED_TEST_SUITE(BandedMatWrappersTest, Implementations);

// =========================================================================
// 1. Test Banded Matrix * Vector Multiplication (y = alpha * A * x + beta * y)
// =========================================================================
TYPED_TEST(BandedMatWrappersTest, BandedMatrixVectorProduct) {
    using T = TypeParam;
    BandedMat<T> A = this->createTestBandedMat();

    std::vector<T> h_x = {1.0, 2.0, 3.0};
    auto x = Vec<T>::create(this->N, this->handle);
    x.set(h_x.data(), this->handle);

    std::vector<T> h_y = {10.0, 10.0, 10.0};
    auto y = Vec<T>::create(this->N, this->handle);
    y.set(h_y.data(), this->handle);

    auto alpha = Singleton<T>::create(static_cast<T>(2.0), this->handle);
    auto beta  = Singleton<T>::create(static_cast<T>(0.5), this->handle);

    std::vector<T> expected = {13, 57.0, 71.0};

    // Invoke bandedMult (takes Handle as pointer)
    A.bandedMult(x, y, &(this->handle), alpha, beta, false);

    // Retrieve results to host
    std::vector<T> actual(this->N);
    y.get(actual.data(), this->handle);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < this->N; ++i) EXPECT_NEAR(actual[i], expected[i], 1e-5);

}

// =========================================================================
// 2. Test Banded Matrix * Dense Matrix Multiplication (Y = alpha * A * X + beta * Y)
// =========================================================================
TYPED_TEST(BandedMatWrappersTest, BandedMatrixDenseMatrixProduct) {
    using T = TypeParam;
    BandedMat<T> A = this->createTestBandedMat();
    const size_t K = 2;

    // Row 0: [1, 4], Row 1: [2, 5], Row 2: [3, 6] -> Column-Major: [1, 2, 3, 4, 5, 6]
    std::vector<T> h_X = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    auto X = Mat<T>::create(this->N, K, this->handle);
    X.set(h_X.data(), this->handle);

    std::vector<T> h_Y = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    auto Y = Mat<T>::create(this->N, K, this->handle);
    Y.set(h_Y.data(), this->handle);

    auto alpha = Singleton<T>::create(static_cast<T>(1.0), this->handle);
    auto beta  = Singleton<T>::create(static_cast<T>(0.0), this->handle);

    // Expected: Y = A * X
    // Expected: Y = A * X using current BandedMat boundary convention
    std::vector<T> expected = {
        4, 26, 33.0,
        13.0, 62.0, 72.0
    };

    A.bandedMult(X, Y, &(this->handle), alpha, beta, false);

    std::vector<T> actual(this->N * K);
    Y.get(actual.data(), this->handle);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1e-5);
    }
}

// =========================================================================
// 3. Test Vector * Banded Matrix Multiplication (y^T = alpha * x^T * A + beta * y^T)
// =========================================================================
TYPED_TEST(BandedMatWrappersTest, VectorBandedMatrixProduct) {
    using T = TypeParam;
    BandedMat<T> A = this->createTestBandedMat();

    std::vector<T> h_x = {1.0, 2.0, 3.0};
    auto x = Vec<T>::create(this->N, this->handle);
    x.set(h_x.data(), this->handle);

    std::vector<T> h_y = {0.0, 0.0, 0.0};
    auto y = Vec<T>::create(this->N, this->handle);
    y.set(h_y.data(), this->handle);

    auto alpha = Singleton<T>::create(static_cast<T>(1.0), this->handle);
    auto beta  = Singleton<T>::create(static_cast<T>(0.0), this->handle);

    // Expected: y^T = x^T * A
    std::vector<T> expected = {8.0, 27.0, 31.0};

    // std::cout << "x = " << GpuOut<T>(x, this->handle) << std::endl;
    // Invoke Vec framework overload (takes Handle as reference)
    x.mult(A, y, this->handle, alpha, beta);

    std::vector<T> actual(this->N);
    y.get(actual.data(), this->handle);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < this->N; ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1e-5);
    }
}

// =========================================================================
// 4. Test Dense Matrix * Banded Matrix Multiplication (Y = alpha * X * A + beta * Y)
// =========================================================================
TYPED_TEST(BandedMatWrappersTest, DenseMatrixBandedMatrixProduct) {
    using T = TypeParam;
    BandedMat<T> A = this->createTestBandedMat();
    const size_t R = 2;

    // Row 0: [1, 2, 3], Row 1: [4, 5, 6] -> Column-Major: [1, 4, 2, 5, 3, 6]
    std::vector<T> h_X = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
    auto X = Mat<T>::create(R, this->N, this->handle);
    X.set(h_X.data(), this->handle);

    // std::cout << "X =\n" << GpuOut<T>(X, this->handle) << std::endl;

    std::vector<T> h_Y = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    auto Y = Mat<T>::create(R, this->N, this->handle);
    Y.set(h_Y.data(), this->handle);

    auto alpha = Singleton<T>::create(static_cast<T>(1.0), this->handle);
    auto beta  = Singleton<T>::create(static_cast<T>(0.0), this->handle);

    // Expected: Y = X * A (Column-Major stored layout)
    std::vector<T> expected = {8.0, 23.0, 27.0, 60.0, 31.0, 67.0};

    auto dense = SquareMat<T>::create(this->N, this->handle);
    A.getDense(dense, this->handle);

    // Invoke Mat framework overload (takes Handle as reference)
    X.mult(A, Y, this->handle, alpha, beta);

    // std::cout << "Y =\n" << GpuOut<T>(Y, this->handle) << std::endl;

    std::vector<T> actual(R * this->N);
    Y.get(actual.data(), this->handle);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1e-5);
    }
}