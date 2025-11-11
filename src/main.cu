#include "deviceArrays/headers/Vec.h"
#include "deviceArrays/headers/SquareMat.h"
#include "deviceArrays/headers/BandedMat.h"
#include "testMethods.cu"
#include "BiCGSTAB.cu"
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <string>
#include <cstring>
#include <functional>
#include <iomanip>
#include <limits>

using namespace std;

inline void checkForDevice(){
    int deviceCount = 0;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found. Exiting.\n";
        std::exit(EXIT_FAILURE);
    }
    std::cout << "CUDA device count: " << deviceCount << "\n";
}

/**
 * @brief Read a file into a device array and print an update for the command line.
 *
 *
 */
template <typename T>
void readAndPrint(GpuArray<T>& array, const string& fileName, const bool isText, Handle& hand) {
    ifstream reader(fileName);
    if(!reader.is_open())
        throw runtime_error("Could not open " + fileName);

    array.set(reader, isText, true, &hand);
    reader.close();

    cout << "Read matrix "<< fileName <<" from file." << endl;
    if(array.size() < 1000) array.get(std::cout << "array = ", true, false, &hand);
    else cout << "The matrix is too large to display." << endl;
}

// Helper function to display the help message.
/**
 * @brief Displays the command-line usage and help information.
 *
 * This function is called when the `-h` flag is passed as a command-line argument.
 * It provides a detailed explanation of the required parameters and their constraints.
 */
void showHelp() {
    cout << "Usage: ./BiCGSTAB [options] <A_file> <num_diags> <matrix_width> <diag_indices> <b_file> <x_dest_file>\n";
    cout << "Options:\n";
    cout << "  -h  Show this help message.\n";
    cout << "  -float     Use single-precision floating-point numbers.\n";
    cout << "  -double    Use double-precision floating-point numbers (default).\n";
    cout << "  -text      Use text files (row-major) instead of binary files (column-major).\n";
    cout << "  -precision <num>   Set the convergence epsilon (default: " << numeric_limits<double>::epsilon() << ").\n";
    cout << "  -max_iter <num>    Set the maximum number of iterations (default 2 * num_rows).\n\n";
    cout << "Arguments:\n";
    cout << "  <A_file>         (1) Path to a file containing the non-zero diagonals of the sparse square matrix 'A'.\n";
    cout << "  <diag_indices>   (2) The number of non 0 diagonals.\n";
    cout << "  <matrix_width>   (3) The height and width of 'A' and the height of b.\n";
    cout << "  <diags_file>     (4) Path to a file containing the indices of the diagonals (space-separated integers).\n";
    cout << "  <b_file>         (5) Path to a file containing the right-hand side vector b.\n";
    cout << "  <x_dest_file>    (6) Path to a destination file for the solution vector x\n";
    cout << "Constraints:\n";
    cout << "  - The files for A, b, and x should contain space-separated floating-point numbers.\n";
    cout << "  - The file for diagonals should contain space-separated integers.\n";
    cout << "  - The dimensions (num_diags, matrix_width) must be positive integers.\n";
    cout << "  - The number of diagonals in diags_file must match num_diags.\n";
}

/**
 * @brief Solves the linear system Ax = b and writes the solution to a file.
 *
 * This function uses the `unpreconditionedBiCGSTAB` algorithm to find the
 * solution vector x and then saves it to the specified destination file.
 *
 * @param A A reference to the CuArray2D containing the matrix A.
 * @param diags A constant reference to the vector of diagonal indices.
 * @param b A reference to the CuArray1D containing the vector b.
 * @param x_dest_file The path to the output file for the solution vector x.
 * @param isText True if the file is a text file, and false if it's a binary file.
 * @param maxIter The maximum number of iterations.
 * @param epsilon A threshold for completion.  How close must a number be to 0 to essentially be 0.
 * @param handle
 * @throws std::runtime_error if the output file cannot be opened.
 */
template <typename T>
void solveAndWriteOutput(BandedMat<T>& A, Vec<T>& b, const string& x_dest_file, const bool isText, size_t maxIter, double epsilon, Handle& handle) {
    Vec<T> x = Vec<T>::create(b.size(), handle);

    Mat<T> space = Mat<T>::create(b.size(), 7);

    BiCGSTAB setup(b, &space, static_cast<T>(epsilon), maxIter);
    setup.solveUnpreconditionedBiCGSTAB(A, x);

    ofstream x_fs(x_dest_file);
    if (!x_fs.is_open()) throw runtime_error("Could not open destination file: " + x_dest_file);

    x.get(x_fs, isText, !isText, &handle);
    x_fs.close();
    cout << "Wrote solution vector x to file: " << x_dest_file << endl;
    if(x.size() < 1000) x.get(std::cout << "x =\n", true, false, &handle);
    else cout << "x is too large to display." << endl;
}

template <typename T>
void solveSystem(int argc, char const* argv[], bool isText, size_t maxIter, double epsilon) {
    string a_file = argv[1];
    size_t numDiags = stoi(argv[2]);
    size_t width = stoi(argv[3]);
    string diags_file = argv[4];
    string b_file = argv[5];
    string x_dest_file = argv[6];

    if (numDiags <= 0 || width <= 0) throw runtime_error("Number of diagonals and matrix width must be positive.");

    Handle hand;

    Vec<T> b = Vec<T>::create(width, hand.stream);
    Vec<int> diags = Vec<int>::create(numDiags, hand.stream);
    BandedMat<T> A = BandedMat<T>::create(width, numDiags, diags);

    readAndPrint(diags, diags_file, isText, hand);
    readAndPrint(A, a_file, isText, hand);

    readAndPrint(b, b_file, isText, hand);

    solveAndWriteOutput(A, b, x_dest_file, isText, maxIter, epsilon, hand);
}

/**
 * @file main.cu
 * @brief This program solves a linear system Ax = b using the unpreconditioned
 * BiCGSTAB method on a CUDA device.
 *
 * It takes the matrix A (stored as packed diagonals), a vector b, and diagonal
 * indices as input from files specified on the command line. The solution x
 * is then written to an output file.
 */
int useCommandLineArgs(int argc, char const* argv[]){
    if (argc == 2) {
        if(string(argv[1]) == "-h")  showHelp();
        return 0;
    }

    if (argc < 7) {
        cerr << "Error: Incorrect number of arguments. Please provide at least the 6 required arguments." << endl;
        showHelp();
        return 1;
    }

    checkForDevice();

    try {
        bool useFloat = false;
        bool isText = false;
        double epsilon = -1.0; // -1.0 indicates default epsilon
        size_t maxIter = std::numeric_limits<size_t>::max();

        // Parse optional flags
        for (size_t i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-float") == 0)
                useFloat = true;

            if (strcmp(argv[i], "-text") == 0)
                isText = true;

            if (strcmp(argv[i], "-precision") == 0) {
                if (i + 1 < argc) {
                    epsilon = stod(argv[i+1]);
                } else {
                    cerr << "Error: '-precision' flag requires a number." << endl;
                    showHelp();
                    return 1;
                }
            }
            if (strcmp(argv[i], "-max_iter") == 0) {
                if (i + 1 < argc) {
                    maxIter = stoi(argv[i+1]);
                } else {
                    cerr << "Error: '-max_iter' flag requires a number." << endl;
                    showHelp();
                    return 1;
                }
            }
        }

        // Set default epsilon if not specified
        if (epsilon < 0)
            epsilon = useFloat ? numeric_limits<float>::epsilon() : numeric_limits<double>::epsilon();

        if (useFloat) solveSystem<float>(argc, argv, isText, maxIter, epsilon);
        else solveSystem<double>(argc, argv, isText, maxIter, epsilon);

    } catch (const exception& e) {
        cerr << "An error occurred: " << e.what() << endl;
        return 1;
    }

    return 0;
}
//TODO: change this class so that it reflects the diagonal -> column nature of BandedMat.

int main(int argc, char const* argv[]) {

    Handle hand;

    size_t denseMatDim = 4;

    auto indices = Vec<int32_t>::create(2);
    indices.get(0).set(1, hand);
    indices.get(1).set(-1, hand);

    auto bm = BandedMat<double>::create(denseMatDim, 2,indices);
    bm.col(0).fill(3, hand);
    bm.col(1).fill(-2, hand);

    auto b = Vec<double>::create(denseMatDim, hand);
    b.fill(1, hand);

    auto x = Vec<double>::create(denseMatDim, hand);

    BiCGSTAB<double> bs(b);

    bs.solveUnpreconditionedBiCGSTAB(bm, x);

    x.get(std::cout << "x = \n", true, false, &hand);

    // return useCommandLineArgs(argc, argv);

}
