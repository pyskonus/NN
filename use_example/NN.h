//
// Created by pyskonus on 6/13/2020.
//

#ifndef USE_EXAMPLE_NN_H
#define USE_EXAMPLE_NN_H

#include "Matrix.h"
#include <vector>
#include <string>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <thread>
#include <fstream>


std::vector<double*> propagate(const std::string& tr_path, std::vector<Matrix>& thetas);
double cost(const std::string& tr_path, const std::string& res_path, const std::vector<Matrix>& thetas);
void backprop(const std::string& tr_path, const std::string& res_path, std::vector<Matrix>& thetas, double learning_rate, size_t iters);
void signalHandler();

#endif //USE_EXAMPLE_NN_H
