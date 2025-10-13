//
// Created by dov on 10/10/25.
//

#ifndef BICGSTAB_BANDEDMAT_H
#define BICGSTAB_BANDEDMAT_H

#include "vec.h"

template<typename T>
class BandedMat final : public Mat<T> {
private:
    const Vec<int32_t> _indices;
protected:
    BandedMat(size_t rows, size_t cols, size_t ld, std::shared_ptr<T> ptr, const Vec<int32_t> &indices);
public:
    BandedMat(const Mat<T>& copyFrom, const Vec<int32_t>& indices);

    static BandedMat create(size_t numDiagonals, size_t cols, const Vec<int32_t> &indices);

    void setFromDense(const SquareMat<T> &denseMat, Handle *handle);

    Vec<T> mult(const Vec<T>& other, Vec<T>* result = nullptr, Handle* handle = nullptr, const Singleton<T> *alpha = nullptr, const Singleton<T> *beta = nullptr, bool transpose = false) const;

    Mat<T> mult(const Mat<T>& other, Mat<T>* result = nullptr, Handle* handle = nullptr, const Singleton<T> *alpha = nullptr, const
                Singleton<T> *beta = nullptr, bool transposeA = false, bool transposeB = false) const;

    void getDense(SquareMat<T> dense, Handle *handle = nullptr) const;


    Mat<T> operator*(const Mat<T>& other) const override;

    Mat<T> plus(const Mat<T>& x, Mat<T>* result = nullptr, const Singleton<T>* alpha = nullptr, const Singleton<T>* beta = nullptr, bool transposeA = false, bool transposeB = false, Handle* handle = nullptr) override;

    Mat<T> minus(const Mat<T>& x, Mat<T>* result = nullptr, const Singleton<T>* alpha = nullptr, const Singleton<T>* beta = nullptr, bool transposeA = false, bool transposeB = false, Handle* handle = nullptr) override;

    void mult(const Singleton<T>& alpha, Handle* handle = nullptr) override;

    void transpose(Mat<T>& result, Handle* handle = nullptr) const override;
    void transpose(Handle* handle = nullptr, Mat<T>* preAlocatedMem = nullptr) override;

    static Mat<T> create(size_t rows, size_t cols);

    [[nodiscard]] Mat<T> subMat(size_t startRow, size_t startCol, size_t height, size_t width) const override;

    Vec<T> col(size_t index) override;
    Vec<T> row(size_t index) override;

    void normalizeCols(size_t setRowTo1, Handle* handle = nullptr) override;
};


#endif //BICGSTAB_BANDEDMAT_H