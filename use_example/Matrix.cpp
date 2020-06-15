#include <thread>
#include "Matrix.h"


double Matrix::vec_vec(const size_t row, const double* vec) const {
    /*auto buffer = new double[2];
    __m128d a;
    __m128d b;
    double result = 0;
    double lower;
    for(size_t i = 0; i < size.second/2; ++i) {
        a = _mm_load_pd(elements+row*size.second+i*2);
        b = _mm_load_pd(vec+i*2);
        __m128d res = _mm_dp_pd(a,b,0xF1);
        _mm_store_pd(buffer, res);
        lower = _mm_cvtsd_f64(res);
        result += lower;
    }
    if(size.second%2==1)
        result += elements[size.second-1]*vec[size.second-1];
    delete[] buffer;
    return result;*/
    return std::inner_product(elements+row*size.second, elements+(row+1)*size.second, vec, 0.0);

}

void Matrix::mat_vec(const double* vec, double* result) const {
    for (size_t i = 0; i < size.first; ++i) {
        result[i] = vec_vec(i, vec);
        /*if(size.first==20)
            std::cout << result[i] << std::endl;*/
    }
}

void Matrix::rand_init() const {
    std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch()/std::chrono::milliseconds(1));
    std::normal_distribution<double> distribution(0,1.0);
    for(size_t i = 0; i < size.first*size.second; ++i) {
        elements[i] = distribution(generator);
    }
}

void Matrix::file_init(const std::string& filename) const {
    std::ifstream infile{filename};
    if(infile.good())
    for(size_t i = 0; i < size.second * size.first; ++i) {
        infile >> elements[i];
    }
    if(!infile.good())
        std::cerr << "there might be an error in the file" << std::endl;
    infile.close();
}

void Matrix::print() const {
    for(size_t i = 0; i < size.first*size.second; ++i) {
        std::cout << elements[i] << " ";
        if((i+1)%size.second==0) {
            std::cout << std::endl;
        }
    }
}

void Matrix::print_file() const {
    std::ofstream ofs{"../data/dummy.dat"};
    for(size_t i = 0; i < size.first*size.second; ++i) {
        ofs << elements[i] << " ";
        if((i+1)%size.second==0) {
            ofs << std::endl;
        }
    }
    ofs.close();
}

void Matrix::print_file(const std::string& filename) const {
    std::ofstream ofs{filename};
    if(!ofs.good())
        std::cout << "bad" << std::endl;
    for(size_t i = 0; i < size.first*size.second; ++i) {
        ofs << elements[i] << " ";
        if((i+1)%size.second==0) {
            ofs << std::endl;
        }
    }
    ofs.close();
}

Matrix Matrix::transpose(double* els) const {
    for(size_t i = 0; i < size.first; ++i)
        for(size_t j = 0; j < size.second; ++j)
            els[j*size.first+i] = elements[i*size.second+j];

    return Matrix{size.second, size.first, els};
}