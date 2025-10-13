
#ifndef BICGSTAB_MAT_H
#define BICGSTAB_MAT_H

#include "deviceArrays.h"

template <typename T>
class Mat : public GpuArray<T> {
    using GpuArray<T>::mult;
protected:
    Mat(size_t rows, size_t cols, size_t ld, std::shared_ptr<T> _ptr);

public:

    [[nodiscard]] size_t size() const override;
    [[nodiscard]] size_t bytes() const override;
    void set(const T* src, cudaStream_t stream) override;
    void get(T* dst, cudaStream_t stream) const override;
    void set(const GpuArray<T>& src, cudaStream_t stream) override;
    void get(GpuArray<T>& dst, cudaStream_t stream) const override;
    void set(std::istream& input_stream, bool isText, bool readColMjr, cudaStream_t stream) override;
    void get(std::ostream& output_stream, bool isText, bool printColMjr, cudaStream_t stream) const override;
    Singleton<T> get(size_t row, size_t col);

    Mat<T> mult(const Mat<T>& other, Mat<T>* result = nullptr, Handle* handle = nullptr, const Singleton<T> *alpha = nullptr, const
                Singleton<T> *beta = nullptr, bool transposeA = false, bool transposeB = false) const;

    virtual Vec<T> mult(const Vec<T>& other, Vec<T>* result, Handle* handle, const Singleton<T> *alpha, const
                        Singleton<T> *beta, bool transpose) const;

    virtual Vec<T> operator*(const Vec<T>& other) const;

    virtual Mat<T> operator*(const Mat<T>& other) const;

    virtual Mat<T> plus(const Mat<T>& x, Mat<T>* result, const Singleton<T>* alpha, const Singleton<T>* beta, bool transposeA, bool transposeB, Handle* handle);

    virtual Mat<T> minus(const Mat<T>& x, Mat<T>* result, const Singleton<T>* alpha, const Singleton<T>* beta , bool transposeA, bool transposeB, Handle* handle);

    virtual void mult(const Singleton<T>& alpha, Handle* handle);

    virtual void transpose(Mat<T>& result, Handle* handle) const;

    virtual void transpose(Handle* handle, Mat<T>* preAlocatedMem);

    static Mat<T> create(size_t rows, size_t cols);

    [[nodiscard]] virtual Mat<T> subMat(size_t startRow, size_t startCol, size_t height, size_t width) const;

    virtual Vec<T> col(size_t index);

    virtual Vec<T> row(size_t index);

    virtual void normalizeCols(size_t setRowTo1, Handle* handle);
};
#endif //BICGSTAB_MAT_H