//TODO: When reading in the diagonals of A, allow for different lengths.

#include "deviceArrays.h"
#include "testMethods.cu"
#include "algorithms.cu"

using namespace std;

/**
 * @brief Read a file into a device array and print an update for the command line.
 * 
 * 
 */
template <typename T>
void readAndPrint(CuArray<T>& array, const string& fileName, const bool isText){
    ifstream reader(fileName);
    if(!reader.is_open()) throw runtime_error("Could not open " + fileName);

    array.set(reader, isText, !isText);
    reader.close();

    cout << "Read matrix "<< fileName <<" from file." << endl;
    if(array.size() < 1000) cout << array << endl;
    else cout << "The matrix is too large to display." << endl;
}

void runAllTests() {
    // Check if a CUDA device is available.
    checkForDevice();
    
    multiTest();

    CuArray2D<float> A(3, 3);
    CuArray1D<float> b(3);
    CuArray1D<int> diags(3);
    CuArray1D<float> x(3);

    float dataA[] = {1.0f, 2.0f, 3.0f,
                     4.0f, 5.0f, 6.0f,
                     0.0f, 8.0f, 0.0f};
    A.set(dataA);

    cout << "A:\n" << A << endl;

    float data_b[] = {1.0f, 2.0f, 3.0f};
    b.set(data_b);

    cout << "b:\n" << b << endl;

    const int data_diags[] = {-1, 0, 1};
    diags.set(data_diags);
    
    
    unpreconditionedBiCGSTAB(A, diags, b, &x, 20, 1e-6f);

    cout << "x:\n" << x << endl;
    cout << "Expected: 2.7500 -1.5000 1.1250 "<< endl;

    cout << "testing multiplication:" << endl;
    float data_x[] = {2.7500f, -1.5000f, 1.1250f};   
    x.set(data_x);
    A.diagMult(diags, x, &b);
    cout << "A*x:\n" << b << "Expected: 1 2 3" << endl;
}

// Helper function to display the help message.
/**
 * @brief Displays the command-line usage and help information.
 *
 * This function is called when the `-h` flag is passed as a command-line argument.
 * It provides a detailed explanation of the required parameters and their constraints.
 */
void showHelp() {
    cout << "Usage: ./unpreconditionedBiCGSTAB [options] <A_file> <num_diags> <matrix_width> <diag_indices> <b_file> <x_dest_file>\n";
    cout << "Options:\n";
    cout << "  -h  Show this help message.\n\n";
    cout << "  -t  Run tests.\n\n";
    cout << "Arguments:\n";
    cout << "  <A_file>         (1) Path to a file containing the non-zero diagonals of the sparse square matrix 'A'.  This file should have a matrix in column major order, where each row is a non 0 diagonal of A.  For diagonals shorter than the primary diagonal, padding is required so that they be the same length.\n";//TODO: make this better.  Matrix A should have a column be a diagonal of A, and shorter diagonals should be padded with 0s to be the same length as the primary diagonal.\n";
    cout << "  <diag_indices>   (2) The number of non 0 diagonals.\n";
    cout << "  <matrix_width>   (3) The height and width of 'A' and the height of b.\n";
    cout << "  <diags_file>     (4) Path to a file containing the indices of the diagonals (space-separated integers).  THe first value should be the index of the first diagonal in the diagonal file, the second value should be the index of the second diagonal, and so on.  Use negative values for sub indices, 0 for the primary diagonal, and positive integer values for the super diagonals. For example, an index of 1 is the superdiagonaladjacent to the primary diagonal.\n";
    cout << "  <b_file>         (5) Path to a file containing the right-hand side vector b.\n";
    cout << "  <x_dest_file>    (6) Path to a destination file for the solution vector x\n";
    cout << "  <optional -text>     Include '-t' as a final optional argument to say that all files are text and not binary files.  If this option is selected, input and output will be row major, and slow.  If this option is not selected, input and output will be column major and faster.\n\n";
    cout << "Constraints:\n";
    cout << "  - The files for A, b, and x should contain space-separated floating-point numbers.\n";
    cout << "  - The file for diagonals should contain space-separated integers.\n";
    cout << "  - The dimensions (num_diags, matrix_width) must be positive integers.\n";
    cout << "  - The number of diagonals in diags_file must match num_diags.\n";
    cout << "  - All files must be in binary and not text format.\n"; //TODO: add flexabuility here.
}

/**
 * @brief Processes the command-line arguments and validates them.
 *
 * @param argc The number of command-line arguments.
 * @param argv The array of command-line argument strings.
 * @param a_file Reference to a string to store the path to the matrix A file.
 * @param numDiags Reference to an int to store the number of diagonals.
 * @param width Reference to an int to store the width of the original matrix.
 * @param diags_file Reference to a string to store the path to the diagonals file.
 * @param b_file Reference to a string to store the path to the vector b file.
 * @param x_dest_file Reference to a string to store the path to the solution x file.
 * @throws std::runtime_error if numDiags or width are not positive.
 */
void processCommandLineArgs(int argc, char const* argv[], string& a_file, int& numDiags, int& width, string& diags_file, string& b_file, string& x_dest_file, bool& isText) {
    a_file = argv[1];
    numDiags = stoi(argv[2]);
    width = stoi(argv[3]);
    diags_file = argv[4];
    b_file = argv[5];
    x_dest_file = argv[6];
    isText = argc == 8 && !strcmp(argv[7], "-text");

    if (numDiags <= 0 || width <= 0) 
        throw runtime_error("Number of diagonals and matrix width must be positive.");    
}

/**
 * @brief Solves the linear system Ax = b and writes the solution to a file.
 *
 * This function uses the `unpreconditionedBiCGSTAB` algorithm to find the
 * solution vector x and then saves it to the specified destination file.
 *
 * @param A A reference to the CuArray2D containing the matrix A.
 * @param diags_vec A constant reference to the vector of diagonal indices.
 * @param b A reference to the CuArray1D containing the vector b.
 * @param x_dest_file The path to the output file for the solution vector x.
 * @throws std::runtime_error if the output file cannot be opened.
 */
void solveAndWriteOutput(CuArray2D<float>& A, const CuArray1D<int>& diags, CuArray1D<float>& b, const string& x_dest_file, const bool isText) {
    CuArray1D<float> x(b.size());
    unpreconditionedBiCGSTAB(A, diags, b, &x, 20, 1e-6f);

    ofstream x_fs(x_dest_file);
    if (!x_fs.is_open()) {
        throw runtime_error("Could not open destination file: " + x_dest_file);
    }
    x.get(x_fs, isText, !isText);
    x_fs.close();
    cout << "Wrote solution vector x to file: " << x_dest_file << endl;
    if(x.size() < 1000) cout << "x:\n" << x << endl;
    else cout << "x is too large to display." << endl;

}

/**
 * @file main.cpp
 * @brief This program solves a linear system Ax = b using the unpreconditioned
 * BiCGSTAB method on a CUDA device.
 *
 * It takes the matrix A (stored as packed diagonals), a vector b, and diagonal
 * indices as input from files specified on the command line. The solution x
 * is then written to an output file.
 */
int main(int argc, char const* argv[]) {
    if (argc == 2) {
        if(string(argv[1]) == "-h")  showHelp();
        if(string(argv[1]) == "-t")  runAllTests();
        return 0;
    }
    else if (argc < 7 || argc > 8) {
        cerr << "Error: Incorrect number of arguments.\n";
        showHelp();
        return 1;
    }

    checkForDevice();

    try {
        string a_file, diags_file, b_file, x_dest_file;
        int numDiags, width;
        bool isText;

        processCommandLineArgs(argc, argv, a_file, numDiags, width, diags_file, b_file, x_dest_file, isText);

        CuArray2D<float> A(numDiags, width);
        CuArray1D<float> b(width);
        CuArray1D<int> diags(numDiags);

        readAndPrint(A, a_file, isText);
        readAndPrint(diags, diags_file, isText);
        readAndPrint(b, b_file, isText);

        solveAndWriteOutput(A, diags, b, x_dest_file, isText);

    } catch (const exception& e) {
        cerr << "An error occurred: " << e.what() << endl;
        return 1;
    }

    return 0;
}
