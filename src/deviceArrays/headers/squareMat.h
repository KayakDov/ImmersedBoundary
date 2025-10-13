

#ifndef BICGSTAB_SQUAREMAT_H
#define BICGSTAB_SQUAREMAT_H

#include "deviceArrays.h"

template <typename T>
class SquareMat : public Mat<T> {
private:
    SquareMat(size_t rowsCols, size_t ld, std::shared_ptr<T> _ptr);

public:
    static SquareMat<T> create(size_t rowsCols);

    void eigen(Vec<T> &eVals, SquareMat<T> *eVecs, Mat<T> *temp = nullptr, Handle *handle = nullptr) const;
};

#endif //BICGSTAB_SQUAREMAT_H