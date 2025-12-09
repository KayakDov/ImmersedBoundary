// #include <iostream>
// #include <vector>
//
// // Thrust library headers for GPU-based vectors and algorithms
// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>
// #include <thrust/sequence.h>
// #include <thrust/transform.h>
// #include <thrust/copy.h>
// #include <thrust/functional.h>
// #include <thrust/iterator/zip_iterator.h>
//
// int main() {
//     // Number of elements in our vectors
//     const int N = 11;
//
//     // --- 1. Create and Initialize Vectors on the GPU (Device) ---
//     // thrust::device_vector is the GPU-side equivalent of std::vector.
//     thrust::device_vector<int> d_ascending(N);
//     thrust::device_vector<int> d_descending(N);
//     thrust::device_vector<int> d_sum(N);
//
//     // Fill the first vector with an ascending sequence from 0 to N-1 (0 to 10)
//     // This is done in parallel on the GPU.
//     thrust::sequence(d_ascending.begin(), d_ascending.end());
//
//     // Fill the second vector with a descending sequence from N-1 to 0 (10 to 0)
//     // We use thrust::transform with a lambda to achieve this.
//     // The lambda takes an index 'i' and returns (N-1) - i.
//     thrust::transform(
//         thrust::make_counting_iterator(0),
//         thrust::make_counting_iterator(N),
//         d_descending.begin(),
//         [=] __device__ (int i) { return (N - 1) - i; }
//     );
//
//     // --- 2. Perform Parallel Summation on the GPU ---
//     // The previous version incorrectly used a unary transform with a binary operator.
//     // The correct way is to use the binary thrust::transform overload, which
//     // takes two input ranges and one output range, along with the binary operator.
//     thrust::transform(d_ascending.begin(), d_ascending.end(),
//                       d_descending.begin(), d_sum.begin(),
//                       thrust::plus<int>());
//
//     // --- 3. Copy Result from GPU to CPU (Host) ---
//     // Create a host_vector to store the results from the GPU.
//     thrust::host_vector<int> h_sum(N);
//     // Copy the contents of the GPU vector 'd_sum' to the CPU vector 'h_sum'.
//     thrust::copy(d_sum.begin(), d_sum.end(), h_sum.begin());
//
//     // --- 4. Print the Results on the CPU ---
//     std::cout << "Ascending vector (on GPU): ";
//     for (int i = 0; i < N; ++i) {
//         // Since the ascending vector is still on the GPU, we can't print it directly.
//         // We can just conceptually know what its values are from the thrust::sequence.
//         std::cout << i << " ";
//     }
//     std::cout << std::endl;
//
//     std::cout << "Descending vector (on GPU): ";
//     for (int i = 0; i < N; ++i) {
//         // Same here, we conceptually know the values.
//         std::cout << (N - 1) - i << " ";
//     }
//     std::cout << std::endl;
//
//     std::cout << "Sum vector (copied to CPU): ";
//     for (int val : h_sum) {
//         std::cout << val << " ";
//     }
//     std::cout << std::endl;
//
//     return 0;
// }
