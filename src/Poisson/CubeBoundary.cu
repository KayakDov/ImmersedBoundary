//
// Created by dov on 11/3/25.
//

#include "deviceArrays/headers/Mat.h"


template<typename T>
class DeviceCubeBoundary {
public:
    const DeviceData2d<T> front, back, left, right, top, bottom;
    __host__ __device__ DeviceCubeBoundary(DeviceData2d<T> front, DeviceData2d<T> back, DeviceData2d<T> left, DeviceData2d<T> right, DeviceData2d<T> top, DeviceData2d<T> bottom):
        front(front), back(back), left(left), right(right), top(top), bottom(bottom) {};
};


template<typename T>
class CubeBoundary {
public:
    /**
     * @brief Boundary Matrix Indexing and Memory Layout for the 3D Grid.
     *
     * This documentation clarifies how the six boundary faces of the 3D grid are mapped into
     * three pairs of 2D matrices, stored sequentially in memory.
     *
     * ---
     *
     * ### 1. Conceptual Alignment and Indexing
     *
     * The six boundary faces are grouped into three matrices: \c frontBack (Front/Back), \c leftRight (Left/Right),
     * and \c topBottom (Top/Bottom).
     *
     * The second matrix in each pair (Back, Right, Bottom) is indexed in perfect **alignment** with the first matrix
     * (Front, Left, Top). For any pair, the $i^{th}$ element of the second matrix corresponds to the element
     * directly "behind" the $i^{th}$ element of the first matrix in the 3D grid.
     *
     * ---
     *
     * ### 2. Physical Orientation (Which Way is Up/Back)
     *
     * This defines the mapping from 2D matrix indices to the 3D grid axes ($\text{X}/\text{cols}, \text{Y}/\text{rows}, \text{Z}/\text{layers}$):
     *
     * * **Vertical Orientation:** The **top** edge (highest row index) of the \c frontBack and \c leftRight matrices corresponds to the **Top face** ($\mathbf{+Y}$ direction) of the 3D grid.
     * * **Depth Orientation:** The **top** edge (highest row index) of the \c topBottom matrix corresponds to the **Back face** ($\mathbf{+Z}$ direction) of the 3D grid.
     *
     * ---
     *
     * ### 3. Memory Layout (Column-Major Stacking)
     *
     * The two faces of each pair are stored sequentially within a single matrix object. The second face (Back, Right, or Bottom) is stored **directly following** the first face (Front, Left, or Top) in memory.
     *
     * This layout assumes that each individual face matrix is stored in **Column-Major order** in memory.
     *
     * \b Example: For a $\mathbf{2 \times 2}$ top and bottom matrix pair stored in a $\mathbf{4 \times 2}$ single matrix:
     * \code
     * topBottom =
     * 1 2  <-- Top Matrix (Rows 0-1)
     * 3 4
     * 5 6  <-- Bottom Matrix (Rows 2-3)
     * 7 8
     * \endcode
     *
     * The individual elements are ordered in column-major storage as:
     * * **Top Matrix (Front Face):** $1 \rightarrow 3 \rightarrow 2 \rightarrow 4$
     * * **Bottom Matrix (Back Face):** $5 \rightarrow 7 \rightarrow 6 \rightarrow 8$
     *
     * Element 5 is conceptually and physically located directly "beneath" element 1, 7 beneath 3, and so on.
     */
    Mat<T> frontBack, leftRight, topBottom;
    /**
     * @brief Constructs the CubeBoundary object by initializing the three boundary matrix pairs.
     *
     * This constructor performs shallow copies of the input matrices, which contain the boundary
     * condition values (e.g., Dirichlet or Neumann values) for the six faces of the 3D grid.
     * The input matrices are structured such that the first half of the rows represents the
     * "Front/Left/Top" faces, and the second half represents the "Back/Right/Bottom" faces.
     *
     * @param frontBack Matrix containing boundary data for the Front (first half of rows) and Back (second half of rows) faces (Y x X plane).
     * @param leftRight Matrix containing boundary data for the Left and Right faces (Y x Z plane).
     * @param topBottom Matrix containing boundary data for the Top and Bottom faces (X x Z plane).
     */
    CubeBoundary(Mat<T>& frontBack, Mat<T>& leftRight, Mat<T>& topBottom):
        frontBack(frontBack), leftRight(leftRight), topBottom(topBottom) {};

    /**
     *
     * @return The size of all the faces.
     */
    [[nodiscard]] size_t size() const {
        return frontBack.size() + leftRight.size() + topBottom.size();
    }

    /**
     *
     * @return The number of elements in the cube surounded by this boundary.
     */
    [[nodiscard]] size_t internalSize() const {
        return frontBack._rows/2 * leftRight._cols * topBottom._rows/2;
    }

    /**
     * A version of this class that is simplified and device friendly.
     * @return
     */
    DeviceCubeBoundary<T> deviceCubeBoundary() const {
        return DeviceCubeBoundary<T>(
            frontBack.subMat(0, 0, frontBack._rows/2, frontBack._cols).toKernel2d(),
            frontBack.subMat(frontBack._rows/2, 0, frontBack._rows/2, frontBack._cols).toKernel2d(),
            leftRight.subMat(0, 0, leftRight._rows/2, leftRight._cols).toKernel2d(),
            leftRight.subMat(leftRight._rows/2, 0, leftRight._rows/2, leftRight._cols).toKernel2d(),
            topBottom.subMat(0, 0, topBottom._rows/2, topBottom._cols).toKernel2d(),
            topBottom.subMat(topBottom._rows/2, 0, topBottom._rows/2, topBottom._cols).toKernel2d()
            );
    }

    /**
     * @brief Calls releaseResources() on an arbitrary number of Mat<T> views.
     *
     * This function is useful for explicitly nullifying the pointers in non-owning
     * Mat<T> objects (views) to prevent their destructors from attempting a
     * double-free, thereby freeing up the memory for the next operation.
     *
     * @tparam T The element type of the matrices.
     * @tparam MatTs A variadic pack of Mat<T> types.
     * @param matrices A reference pack of Mat<T> objects (views).
     */
    void freeMem() {
        frontBack.freeMem();
        leftRight.freeMem();
        topBottom.freeMem();
    }


    /**
     *  generates a sample boundary condition.  The front face is 1, the back face is 0, and the other faces linearly
     *  transition from front value to back value.
     * @param dimLength The length of each side of the cube.
     * @return A cube boundary.
     */
    static CubeBoundary<T> ZeroTo1(const size_t dimLength, cudaStream_t stream) {
        const size_t height = dimLength, width = dimLength, depth = dimLength;
        constexpr T frontFaceVal = 1;

        auto boundaries = Mat<T>::create(
            height * 4 + depth * 2,
            std::max(depth, width)
            );

        auto frontBack = boundaries.subMat(0, 0, 2*height, width),
            leftRight = boundaries.subMat(frontBack._rows, 0, 2 * height, depth),
            topBottom = boundaries.subMat(frontBack._rows + leftRight._rows, 0, 2 * depth, width);

        auto front = frontBack.subMat(0, 0, height, width),
            back = frontBack.subMat(height, 0, height, width),
            top = topBottom.subMat(0, 0, depth, width),
            bottom = topBottom.subMat(depth, 0, depth, width);

        front.fill(frontFaceVal, stream);
        back.fill(0, stream);

        for (size_t layerInd = 0; layerInd < depth; ++layerInd) {
            T val = frontFaceVal * (static_cast<T>(layerInd) + static_cast<T>(1))/(depth + static_cast<T>(1));
            leftRight.col(layerInd).fill(val, stream);
            top.row(layerInd).fill(val, stream);
            bottom.row(layerInd).fill(val, stream);
        }

        return CubeBoundary<T>(frontBack, leftRight, topBottom);
    }
};
