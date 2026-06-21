#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// Framework Headers (Adjust paths based on your project structure)
#include "headers/sparse/BandedMat.h"
#include "headers/Mat.h"
#include "headers/Vec.h"
#include "headers/Singleton.h"

template <typename T>
class BandedMatWrappersTest : public ::testing::Test {
protected:
    Handle handle;
    
    // Matrix dimensions
    const size_t N = 3;
    const size_t numDiags = 3;

    // Host data arrays
    std::vector<int32_t> h_diags = {-1, 0, 1}; // Sub, Main, Super
    
    // Banded matrix stored in column-major order matching your framework structure
    // Col 0 (diag -1): [3, 6, x] -> padded/garbage at trailing
    // Col 1 (diag  0): [2, 4, 7]
    // Col 2 (diag  1): [x, 1, 5] -> padded/garbage at leading
    std::vector<T> h_banded_data = {
        3.0, 6.0, 0.0,  // Diag -1
        2.0, 4.0, 7.0,  // Diag  0
        0.0, 1.0, 5.0   // Diag  1
    };

    // Device allocations
    T *d_banded_data = nullptr;
    int32_t *d_diags = nullptr;

    void SetUp() override {
        cudaMalloc(&d_banded_data, h_banded_data.size() * sizeof(T));
        cudaMalloc(&d_diags, h_diags.size() * sizeof(int32_t));

        cudaMemcpy(d_banded_data, h_banded_data.data(), h_banded_data.size() * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(d_diags, h_diags.data(), h_diags.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    }

    void TearDown() override {
        if (d_banded_data) cudaFree(d_banded_data);
        if (d_diags) cudaFree(d_diags);
    }

    // Helper to instantiate BandedMat object using framework factory methods
    BandedMat<T> createTestBandedMat() {
        return BandedMat<T>::create(N, numDiags, N, d_banded_data, d_diags, 1);
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

    // Setup input vector x = [1, 2, 3]^T
    std::vector<T> h_x = {1.0, 2.0, 3.0};
    auto x = Vec<T>::create(this->N, 1, h_x.data()); // Assuming non-owning or managed device copy wrapper

    // Setup initial destination vector y = [10, 10, 10]^T
    std::vector<T> h_y = {10.0, 10.0, 10.0};
    auto y = Vec<T>::create(this->N, 1, h_y.data());

    // Scalars: alpha = 2.0, beta = 0.5
    auto alpha = Singleton<T>::create(static_cast<T>(2.0));
    auto beta = Singleton<T>::create(static_cast<T>(0.5));

    // Expected mathematical result calculation:
    // Ax = [ (2*1 + 1*2), (3*1 + 4*2 + 5*3), (6*2 + 7*3) ]^T = [4, 26, 33]^T
    // y = 2.0 * Ax + 0.5 * y_old = [ 2*4 + 5, 2*26 + 5, 2*33 + 5 ]^T = [13, 57, 71]^T
    std::vector<T> expected = {13.0, 57.0, 71.0};

    // Invoke host wrapper
    A.bandedMult(x, y, &(this->handle), alpha, beta, false);
    cudaDeviceSynchronize();

    // Verify back on host
    std::vector<T> actual(this->N);
    cudaMemcpy(actual.data(), y.toKernel1d().data, this->N * sizeof(T), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < this->N; ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1e-5);
    }
}

// =========================================================================
// 2. Test Banded Matrix * Dense Matrix Multiplication (Y = alpha * A * X + beta * Y)
// =========================================================================
TYPED_TEST(BandedMatWrappersTest, BandedMatrixDenseMatrixProduct) {
    using T = TypeParam;
    BandedMat<T> A = this->createTestBandedMat();
    const size_t K = 2; // Number of columns in dense matrix X

    // Input Dense Matrix X (3x2, Column-Major layout implied by framework)
    // Row 0: [1, 4], Row 1: [2, 5], Row 2: [3, 6] -> Data: [1, 2, 3, 4, 5, 6]
    std::vector<T> h_X = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    auto X = Mat<T>::create(this->N, K, this->N, h_X.data());

    // Initial Destination Matrix Y (3x2, initialized to 0.0, beta = 0.0)
    std::vector<T> h_Y = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    auto Y = Mat<T>::create(this->N, K, this->N, h_Y.data());

    auto alpha = Singleton<T>::create(static_cast<T>(1.0));
    auto beta = Singleton<T>::create(static_cast<T>(0.0));

    // Expected Mathematical Result Calculation (A * X):
    // Col 0: A * [1, 2, 3]^T = [4, 26, 33]^T
    // Col 1: A * [4, 5, 6]^T = [ (2*4 + 1*5), (3*4 + 4*5 + 5*6), (6*5 + 7*6) ]^T = [13, 62, 72]^T
    std::vector<T> expected = {4.0, 26.0, 33.0, 13.0, 62.0, 72.0};

    // Invoke host wrapper
    A.bandedMult(X, Y, &(this->handle), alpha, beta, false);
    cudaDeviceSynchronize();

    std::vector<T> actual(this->N * K);
    cudaMemcpy(actual.data(), Y.toKernel2d().data, actual.size() * sizeof(T), cudaMemcpyDeviceToHost);

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

    // Vector x^T = [1, 2, 3]
    std::vector<T> h_x = {1.0, 2.0, 3.0};
    auto x = Vec<T>::create(this->N, 1, h_x.data());

    // Destination vector y^T initialized to zeros
    std::vector<T> h_y = {0.0, 0.0, 0.0};
    auto y = Vec<T>::create(this->N, 1, h_y.data());

    auto alpha = Singleton<T>::create(static_cast<T>(1.0));
    auto beta = Singleton<T>::create(static_cast<T>(0.0));

    // Expected Mathematical Result Calculation (x^T * A):
    // [1, 2, 3] * A = [ (1*2 + 2*3), (1*1 + 2*4 + 3*6), (2*5 + 3*7) ] = [8, 27, 31]
    std::vector<T> expected = {8.0, 27.0, 31.0};

    // Invoke wrapper from Vec class
    x.productVecBanded(A, y, &(this->handle), &alpha, &beta);
    cudaDeviceSynchronize();

    std::vector<T> actual(this->N);
    cudaMemcpy(actual.data(), y.toKernel1d().data, this->N * sizeof(T), cudaMemcpyDeviceToHost);

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
    const size_t R = 2; // Number of rows in dense matrix X

    // Input Dense Matrix X (2x3, Column-Major Layout)
    // Row 0: [1, 2, 3], Row 1: [4, 5, 6] -> Strided storage: [1, 4, 2, 5, 3, 6]
    std::vector<T> h_X = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
    auto X = Mat<T>::create(R, this->N, R, h_X.data());

    // Destination matrix Y (2x3, Column-Major Layout)
    std::vector<T> h_Y = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    auto Y = Mat<T>::create(R, this->N, R, h_Y.data());

    auto alpha = Singleton<T>::create(static_cast<T>(1.0));
    auto beta = Singleton<T>::create(static_cast<T>(0.0));

    // Expected Mathematical Result Calculation (X * A):
    // Row 0: [1, 2, 3] * A = [8, 27, 31]
    // Row 1: [4, 5, 6] * A = [ (4*2 + 5*3), (4*1 + 5*4 + 6*6), (5*5 + 6*7) ] = [23, 60, 67]
    // Stored Column-Major: [Row0_Col0, Row1_Col0, Row0_Col1, Row1_Col1, Row0_Col2, Row1_Col2]
    std::vector<T> expected = {8.0, 23.0, 27.0, 60.0, 31.0, 67.0};

    // Invoke wrapper from Mat class
    X.productMatBanded(A, Y, &(this->handle), &alpha, &beta);
    cudaDeviceSynchronize();

    std::vector<T> actual(R * this->N);
    cudaMemcpy(actual.data(), Y.toKernel2d().data, actual.size() * sizeof(T), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1e-5);
    }
}