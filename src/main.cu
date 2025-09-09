#include "deviceArrays.h"
#include "testMethods.cu"
#include "algorithms.cu"
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <string>
#include <cstring>
#include <iomanip>
#include <limits>

using namespace std;

/**
 * @brief Read a file into a device array and print an update for the command line.
 *
 *
 */
template <typename T>
void readAndPrint(GpuArray<T>& array, const string& fileName, const bool isText) {
    ifstream reader(fileName);
    if(!reader.is_open())
        throw runtime_error("Could not open " + fileName);

    array.set(reader, isText, !isText);
    reader.close();

    cout << "Read matrix "<< fileName <<" from file." << endl;
    if(array.size() < 1000)
        cout << array << endl;
    else
        cout << "The matrix is too large to display." << endl;
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
 * @throws std::runtime_error if the output file cannot be opened.
 */
template <typename T>
void solveAndWriteOutput(Mat<T>& A, const Vec<int>& diags, Vec<T>& b, const string& x_dest_file, const bool isText, int maxIter, double epsilon) {
    Vec<T> x(b.size());
    BiCGSTAB setup(b, static_cast<T>(epsilon), maxIter);
    setup.unpreconditionedBiCGSTAB(A, diags, &x);

    ofstream x_fs(x_dest_file);
    if (!x_fs.is_open())
        throw runtime_error("Could not open destination file: " + x_dest_file);

    x.get(x_fs, isText, !isText);
    x_fs.close();
    cout << "Wrote solution vector x to file: " << x_dest_file << endl;
    if(x.size() < 1000)
        cout << "x:\n" << x << endl;
    else
        cout << "x is too large to display." << endl;
}

template <typename T>
void solveSystem(int argc, char const* argv[], bool isText, int maxIter, double epsilon) {
    string a_file = argv[1];
    int numDiags = stoi(argv[2]);
    int width = stoi(argv[3]);
    string diags_file = argv[4];
    string b_file = argv[5];
    string x_dest_file = argv[6];

    if (numDiags <= 0 || width <= 0)
        throw runtime_error("Number of diagonals and matrix width must be positive.");

    // Check if maxIter needs to be set to its default
    if (maxIter == -1)
        maxIter = width * 2;

    Mat<T> A(numDiags, width);
    Vec<T> b(width);
    Vec<int> diags(numDiags);

    readAndPrint(A, a_file, isText);
    readAndPrint(diags, diags_file, isText); 
    readAndPrint(b, b_file, isText);

    solveAndWriteOutput(A, diags, b, x_dest_file, isText, maxIter, epsilon);
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
        int maxIter = -1; // -1 indicates default max iterations

        // Parse optional flags
        for (int i = 1; i < argc; ++i) {
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


int main(int argc, char const* argv[]) {
    
    return useCommandLineArgs(argc, argv);

}
