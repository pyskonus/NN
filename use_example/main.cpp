//
// Created by pyskonus on 6/13/2020.
//

#include "Matrix.h"
#include "NN.h"
#include <random>
#include <iostream>
#include <chrono>
#include <vector>
#include <mutex>
#include <condition_variable>


std::condition_variable condom;
std::mutex muteks;

int main() {
    size_t thetas_am = 3;
    std::vector<std::pair<size_t, size_t>> sizes(thetas_am);
    std::vector<double*> elementss(thetas_am);
    std::vector<Matrix> thetas(thetas_am);
    sizes[0] = std::pair<size_t, size_t>(100, 1000);
    sizes[1] = std::pair<size_t, size_t>(20, 101);
    sizes[2] = std::pair<size_t, size_t>(4, 21);
    for(size_t i = 0; i < thetas_am; ++i) {
        elementss[i] = new double[sizes[i].first*sizes[i].second];
        thetas[i] = Matrix{sizes[i].first, sizes[i].second, elementss[i]};
    }
    /*thetas[0].file_init("../data/theta1.dat");
    thetas[1].file_init("../data/theta2.dat");
    thetas[2].file_init("../data/theta3.dat");*/
    thetas[0].file_init("../data/theta1r.dat");
    thetas[1].file_init("../data/theta2r.dat");
    thetas[2].file_init("../data/theta3r.dat");
    /*thetas[0].rand_init();
    thetas[1].rand_init();
    thetas[2].rand_init();*/
    /*thetas[0].file_init("../data/MNIST/Theta1.dat");
    thetas[1].file_init("../data/MNIST/Theta2.dat");*/

    std::string tr_path = "../data/tr_ex.dat";
    std::string res_path = "../data/res.dat";
    /*auto result = propagate(tr_path, thetas);
    std::ofstream outfile{"../data/MNIST/my_y.dat"};
    for(const auto& row: result) {
        for(size_t i = 0; i < 10; ++i) {
            outfile << row[i] << " ";
        }
        outfile << std::endl;
    }
    outfile.close();*/
    /*std::cout << cost(tr_path, res_path, thetas) << std::endl;
    backprop(tr_path, res_path, thetas, 0.001, 10);
    std::cout << cost(tr_path, res_path, thetas);
    /// write out
    std::vector<std::string> filenames(thetas.size());
    for(size_t i = 0; i < thetas.size(); ++i) {
        filenames[i] = "../data/"+std::to_string(i+1)+".dat";
        thetas[i].print_file(filenames[i]);
    }*/

    thetas[2].print_file();
    backprop(tr_path, res_path, thetas, 0.001, 10);

    for(auto& els: elementss) {
        delete[] els;
    }

    return 0;
}