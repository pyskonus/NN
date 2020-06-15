#ifndef USE_EXAMPLE_MATRIX_H
#define USE_EXAMPLE_MATRIX_H

#include <utility>
#include <cstddef>
#include <iostream>
#include <smmintrin.h>
#include <random>
#include <chrono>
#include <fstream>


class Matrix {
public:
    std::pair<size_t, size_t> size;
    double* elements;

    Matrix() = default;
    Matrix(size_t first, size_t second, double* elements_): size{first, second}
        {elements=elements_;}
    double vec_vec(size_t row, const double* vec) const;
    void mat_vec(const double* vec, double* result) const;
    void rand_init() const;
    void file_init(const std::string& filename) const;
    void print() const;
    void print_file() const;
    void print_file(const std::string& filename) const;
    Matrix transpose(double* els) const;
    /*~Matrix() {delete[] elements;}*/  /// actually this should have been
    /// the destructor, but it is too dangerous to refactor at this point
    ~Matrix() = default;
};

#endif //USE_EXAMPLE_MATRIX_H
