
#ifndef BICGSTAB_VEC_H
#define BICGSTAB_VEC_H

#include "Mat.h"
#include "Tensor.h"

template <typename T>
class Vec : public GpuArray<T> {
    using GpuArray<T>::mult;
private:
    friend Vec<T> Mat<T>::row(size_t index);
    friend Vec<T> Mat<T>::col(size_t index);
    friend Vec<T> Tensor<T>::depth(size_t row, size_t col);
protected:
    Vec(size_t cols, std::shared_ptr<T> _ptr, size_t stride);
public:

    static Vec<T> create(size_t length, cudaStream_t stream = nullptr);

    Vec<T> subVec(size_t offset, size_t length, size_t stride = 1) const;

    [[nodiscard]] size_t size() const override;
    [[nodiscard]] size_t bytes() const override;
    void set(const T* hostData, cudaStream_t stream) override;
    void get(T* hostData, cudaStream_t stream) const override;
    void set(const GpuArray<T>& src, cudaStream_t stream) override;
    void get(GpuArray<T>& dst, cudaStream_t stream) const override;
    void set(std::istream& input_stream, bool isText, bool isColMjr, cudaStream_t stream) override;
    void get(std::ostream& output_stream, bool isText, bool isColMjr, cudaStream_t stream) const override;

    void fill(T val, cudaStream_t stream) override;

    Singleton<T> get(size_t i);

    Vec<T> mult(const Mat<T>& other, Vec<T>* result = nullptr, Handle* handle = nullptr, const Singleton<T>* alpha = nullptr, const Singleton<T>* beta = nullptr, bool transpose = false) const;

    T mult(const Vec<T>& other, Singleton<T>* result = nullptr, Handle* handle = nullptr) const;

    Vec<T> operator*(const Mat<T>& other) const;
    T operator*(const Vec<T>& other) const;


    void add(const Vec<T>& x, const Singleton<T> *alpha, Handle* handle);
    void sub(const Vec<T>& x, const Singleton<T>* alpha, Handle* handle);

    void mult(const Singleton<T>& alpha, Handle* handle = nullptr);

    void fillRandom(Handle* handle = nullptr);

    void EBEPow(const Singleton<T>& t, const Singleton<T>& n, cudaStream_t stream);

    void setSum(const Vec& a, const Vec& B, const Singleton<T>* alpha, const Singleton<T>* beta, Handle* handle);

};

#endif //BICGSTAB_VEC_H