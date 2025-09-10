// #include <iostream>
// #include <vector>
// #include <string>
// #include <numeric>
// #include <typeinfo>
// #include <type_traits>
// #include <fstream>
// #include <cstdio>
// #include <cmath>
// #include <stdexcept>
//
// #include "deviceArrays.h"  // Assuming this file exists and is correct
//
// // ---------- Utilities ----------
//
// template <typename T>
// static void printMismatch(const std::vector<T>& expected,
//                           const std::vector<T>& result,
//                           const std::string& test_name,
//                           size_t max_show = 10)
// {
//     std::cerr << "❌ " << test_name << " failed.\n";
//     std::cerr << "Expected size: " << expected.size()
//               << ", Result size: "  << result.size() << "\n";
//     size_t n = std::min(std::min(expected.size(), result.size()), max_show);
//     for (size_t i = 0; i < n; ++i) {
//         std::cerr << "  [" << i << "] exp=" << expected[i]
//                   << " got=" << result[i] << "\n";
//     }
// }
//
// template <typename T>
// void verifyVectors(const std::vector<T>& expected,
//                    const std::vector<T>& result,
//                    const std::string& test_name)
// {
//     if constexpr (std::is_floating_point<T>::value) {
//         if (expected.size() != result.size()) {
//             printMismatch(expected, result, test_name);
//             return;
//         }
//         // Tolerance scaled for floats/doubles
//         const double tol = std::is_same<T,float>::value ? 1e-4 : 1e-10;
//         bool ok = true;
//         for (size_t i = 0; i < expected.size(); ++i) {
//             double a = static_cast<double>(expected[i]);
//             double b = static_cast<double>(result[i]);
//             if (std::abs(a - b) > tol * std::max(1.0, std::abs(a))) {
//                 ok = false; break;
//             }
//         }
//         if (ok) {
//             std::cout << "✅ " << test_name << " successful.\n";
//         } else {
//             printMismatch(expected, result, test_name);
//         }
//     } else {
//         if (expected == result) {
//             std::cout << "✅ " << test_name << " successful.\n";
//         } else {
//             printMismatch(expected, result, test_name);
//         }
//     }
// }
//

//
// // ---------- Tests ----------
//
// template <typename T>
// void runBasicArrayTests()
// {
//     std::cout << "--- Running basic array tests for type " << typeid(T).name() << " ---\n";
//
//     // ---- 2D set/get ----
//     {
//         const size_t rows = 4, cols = 3;
//         std::vector<T> h(rows * cols);
//         // Fill column-major: col c contains {c*10 + 1, c*10 + 2, ...}
//         for (size_t c = 0; c < cols; ++c)
//             for (size_t r = 0; r < rows; ++r)
//                 h[c*rows + r] = static_cast<T>(c*10 + r + 1);
//
//         Mat<T> d(rows, cols);
//         d.set(h.data());
//         CHECK_CUDA_ERROR(cudaDeviceSynchronize());
//
//         std::vector<T> out(rows * cols);
//         d.get(out.data());
//         CHECK_CUDA_ERROR(cudaDeviceSynchronize());
//         verifyVectors(h, out, "CuArray2D set/get");
//     }
//
//     // ---- 2D subarray view ----
//     {
//         const size_t rows = 4, cols = 3;
//         std::vector<T> h(rows * cols);
//         for (size_t c = 0; c < cols; ++c)
//             for (size_t r = 0; r < rows; ++r)
//                 h[c*rows + r] = static_cast<T>(100 + c*10 + r); // unique
//
//         Mat<T> parent(rows, cols);
//         parent.set(h.data());
//         CHECK_CUDA_ERROR(cudaDeviceSynchronize());
//
//         const size_t startRow = 1, startCol = 1;
//         const size_t subH = 2, subW = 2;
//         Mat<T> sub(parent, startRow, startCol, subH, subW);
//
//         std::vector<T> got(subH * subW);
//         sub.get(got.data());
//         CHECK_CUDA_ERROR(cudaDeviceSynchronize());
//
//         std::vector<T> expected;
//         expected.reserve(got.size());
//         for (size_t c = 0; c < subW; ++c)
//             for (size_t r = 0; r < subH; ++r)
//                 expected.push_back(h[(startCol + c)*rows + (startRow + r)]);
//
//         verifyVectors(expected, got, "CuArray2D subarray view");
//     }
//
//     // ---- 1D set/get ----
//     {
//         const size_t n = 9;
//         std::vector<T> h(n);
//         for (size_t i = 0; i < n; ++i) h[i] = static_cast<T>(i + 5);
//
//         Vec<T> d(n);
//         d.set(h.data());
//         CHECK_CUDA_ERROR(cudaDeviceSynchronize());
//
//         std::vector<T> out(n);
//         d.get(out.data());
//         CHECK_CUDA_ERROR(cudaDeviceSynchronize());
//         verifyVectors(h, out, "CuArray1D set/get");
//     }
//
//     // ---- 1D extraction from 2D (row & column) ----
//     {
//         const size_t rows = 5, cols = 4;
//         std::vector<T> h(rows * cols);
//         for (size_t c = 0; c < cols; ++c)
//             for (size_t r = 0; r < rows; ++r)
//                 h[c*rows + r] = static_cast<T>(c*100 + r);
//
//         Mat<T> A(rows, cols);
//         A.set(h.data());
//         CHECK_CUDA_ERROR(cudaDeviceSynchronize());
//
//         // Column extract
//         int col = 2;
//         Vec<T> d_col(A, col, IndexType::Column);
//         std::vector<T> got_col(rows);
//         d_col.get(got_col.data());
//         CHECK_CUDA_ERROR(cudaDeviceSynchronize());
//         std::vector<T> exp_col(rows);
//         for (size_t r = 0; r < rows; ++r) exp_col[r] = h[col*rows + r];
//         verifyVectors(exp_col, got_col, "CuArray1D column extraction");
//
//         // Row extract
//         int row = 3;
//         Vec<T> d_row(A, row, IndexType::Row);
//         std::vector<T> got_row(cols);
//         d_row.get(got_row.data());
//         CHECK_CUDA_ERROR(cudaDeviceSynchronize());
//         std::vector<T> exp_row(cols);
//         for (size_t c = 0; c < cols; ++c) exp_row[c] = h[c*rows + row];
//         verifyVectors(exp_row, got_row, "CuArray1D row extraction");
//     }
// }
//
// // Multiplication tests only for floating-point (API provides float/double overloads)
// template <typename T>
// void runMulTests()
// {
//     static_assert(std::is_floating_point<T>::value, "Mul tests require float/double");
//
//     std::cout << "--- Running mul tests for " << typeid(T).name() << " ---\n";
//
//     // A (m x k) * B (k x n) = C (m x n)
//     const int m = 2, k = 3, n = 2;
//
//     // Define A and B in NORMAL math form (row-major presentation) for clarity:
//     // A = [ [1, 2, 3],
//     //       [4, 5, 6] ]  (2x3)
//     // B = [ [7,  8],
//     //       [9, 10],
//     //       [11,12] ]    (3x2)
//     //
//     // Now store them in COLUMN-MAJOR order for the device arrays:
//     std::vector<T> hA = {
//         T(1), T(4),  // col 0
//         T(2), T(5),  // col 1
//         T(3), T(6)   // col 2
//     }; // size m*k
//
//     std::vector<T> hB = {
//         T(7),  T(9),  T(11), // col 0 (k entries)
//         T(8),  T(10), T(12)  // col 1
//     }; // size k*n
//
//     std::vector<T> hC_expected = {
//         T(58), T(139),  // column 0 of C: [58,139]^T
//         T(64), T(154)   // column 1 of C: [64,154]^T
//     }; // column-major (m*n entries)
//
//     // Load to device
//     Mat<T> dA(m, k);
//     Mat<T> dB(k, n);
//
//     dA.set(hA.data());
//     dB.set(hB.data());
//     CHECK_CUDA_ERROR(cudaDeviceSynchronize());
//
//     // Test operator* (returns a new CuArray2D<T>)
//     auto dC_new = dA * dB;
//     std::vector<T> hC(m*n);
//     dC_new.get(hC.data());
//     CHECK_CUDA_ERROR(cudaDeviceSynchronize());
//     verifyVectors(hC_expected, hC, "CuArray2D operator* GEMM");
//
//     // Test out-parameter version (mult into pre-allocated)
//     Mat<T> dC(m, n);
//     dA.mult(dB, &dC);
//     std::vector<T> hC2(m*n);
//     dC.get(hC2.data());
//     CHECK_CUDA_ERROR(cudaDeviceSynchronize());
//     verifyVectors(hC_expected, hC2, "CuArray2D mult (into result)");
//
//     // Matrix-Vector: y = A * x, x length = k, y length = m
//     std::vector<T> hx = { T(1), T(1), T(1) }; // 3
//     std::vector<T> hy_expected = { T(6), T(15) }; // [1+2+3, 4+5+6]
//
//     Vec<T> dx(k);
//     dx.set(hx.data());
//     CHECK_CUDA_ERROR(cudaDeviceSynchronize());
//
//     // operator* returns a CuArray1D<T>
//     Vec<T> dy_new = dA * dx;
//     std::vector<T> hy(m);
//     dy_new.get(hy.data());
//     CHECK_CUDA_ERROR(cudaDeviceSynchronize());
//     verifyVectors(hy_expected, hy, "CuArray2D * CuArray1D (gemv)");
//
//     // Fix: Remove the redundant instantiation and directly assign
//     Vec<T> dy2 = dA.mult(dx);
//     std::vector<T> hy2(m);
//     dy2.get(hy2.data());
//     CHECK_CUDA_ERROR(cudaDeviceSynchronize());
//     verifyVectors(hy_expected, hy2, "CuArray2D.mult(vec) (gemv)");
//
//     // Vector-Vector dot: use operator*
//     std::vector<T> hv1 = { T(1), T(2), T(3) };
//     std::vector<T> hv2 = { T(4), T(5), T(6) };
//     Vec<T> dv1(hv1.size()), dv2(hv2.size());
//     dv1.set(hv1.data());
//     dv2.set(hv2.data());
//     CHECK_CUDA_ERROR(cudaDeviceSynchronize());
//
//     T dot = dv1 * dv2; // operator* for dot
//     T expected_dot = T(32);
//     if constexpr (std::is_same<T,float>::value) {
//         if (std::abs(dot - expected_dot) <= T(1e-4))
//             std::cout << "✅ CuArray1D dot product successful.\n";
//         else
//             std::cout << "❌ CuArray1D dot product failed. Expected "
//                       << expected_dot << " got " << dot << "\n";
//     } else {
//         if (dot == expected_dot)
//             std::cout << "✅ CuArray1D dot product successful.\n";
//         else
//             std::cout << "❌ CuArray1D dot product failed. Expected "
//                       << expected_dot << " got " << dot << "\n";
//     }
// }
//
// template <typename T>
// void runFileIOTests()
// {
//     static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
//                   "File I/O tests require arithmetic types");
//
//     std::cout << "\n--- Running File I/O tests for type " << typeid(T).name() << " ---\n";
//
//     // Use non-trivial pitch to exercise 2D copies (rows not multiple of 128 bytes etc.)
//     const size_t rows = 513;
//     const size_t cols = 257;
//     const size_t N    = rows * cols;
//
//     // Fill column-major
//     std::vector<T> h(N);
//     for (size_t c = 0; c < cols; ++c)
//         for (size_t r = 0; r < rows; ++r)
//             h[c*rows + r] = static_cast<T>(c*3 + r % 17);
//
//     std::string fname = "test_array.bin";
//     try {
//         Mat<T> d(rows, cols);
//         d.set(h.data());
//         CHECK_CUDA_ERROR(cudaDeviceSynchronize());
//
//         // Write to file via stream API
//         {
//             std::ofstream out(fname, std::ios::binary);
//             if (!out) throw std::runtime_error("Could not open file for writing.");
//             d.get(out);
//         }
//         CHECK_CUDA_ERROR(cudaDeviceSynchronize());
//
//         // Read back into a fresh array
//         Mat<T> d2(rows, cols);
//         {
//             std::ifstream in(fname, std::ios::binary);
//             if (!in) throw std::runtime_error("Could not open file for reading.");
//             d2.set(in);
//         }
//         CHECK_CUDA_ERROR(cudaDeviceSynchronize());
//
//         // Verify
//         std::vector<T> h2(N);
//         d2.get(h2.data());
//         CHECK_CUDA_ERROR(cudaDeviceSynchronize());
//         verifyVectors(h, h2, "File I/O roundtrip");
//
//     } catch (const std::exception& e) {
//         std::cerr << "File I/O Test Failed: " << e.what() << "\n";
//     }
//
//     std::remove(fname.c_str());
// }
//
// // ---------- Public entry points declared in the header ----------
//
// template <typename T>
// void runTests()
// {
//     checkForDevice();
//     runBasicArrayTests<T>();
//
//     // Only float/double have GEMM/GEMV/dot overloads in your API
//     if constexpr (std::is_same<T,float>::value || std::is_same<T,double>::value) {
//         runMulTests<T>();
//     }
// }
//
// void inline multiTest()
// {
//     checkForDevice();
//
//     std::cout << "\n===== Basic (int) tests =====\n";
//
//     std::cout << "\n===== Float tests =====\n";
//     runTests<float>();   // includes GEMM/GEMV/dot
//
//     std::cout << "\n===== Double tests =====\n";
//     runTests<double>();  // includes GEMM/GEMV/dot
//
//     std::cout << "\n===== File I/O (int) =====\n";
//     runFileIOTests<int>();
//
//     std::cout << "\n===== File I/O (float) =====\n";
//     runFileIOTests<float>();
//
//     std::cout << "\n===== File I/O (float) =====\n";
//     runFileIOTests<double>();
// }
//
// // ---------- Explicit instantiations (optional but harmless) ----------
//
// template void runTests<int>();
// template void runTests<float>();
// template void runTests<double>();
// template void runFileIOTests<int>();
// template void runFileIOTests<float>();
