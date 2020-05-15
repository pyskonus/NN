#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "libfiles/NNetwork.h"

int main() {

    std::vector<std::vector<double>> X_ = std::vector{std::vector<double>{1,2,3,4}, std::vector<double>{2,3,4,5}, std::vector<double>{3,4,5,6}};
    std::vector<std::vector<double>> y_ = std::vector{std::vector<double>{1,2}, std::vector<double>{2,3}, std::vector<double>{3, 4}};
    NNetwork nn1 = NNetwork(X_);
    Eigen::MatrixXd temp = NNetwork::load_file("../matrices/X.dat");
    NNetwork nn = NNetwork(temp);   /// 100x2

    Eigen::MatrixXd mtx = Eigen::MatrixXd(1, 3);
    mtx << 20, 40, -30;
    auto thetas_ = std::vector{mtx};
    nn.set_thetas(thetas_);
    auto output = nn.feed_forward();

    std::ofstream res{"../matrices/obtained.dat"};
    res << output << std::endl;
    res.close();
    auto temp_y = NNetwork::load_file("../matrices/y.dat");
    nn.set_y(temp_y);
    std::cout << nn.cost() << std::endl;
    std::cout << nn.cost_reg(0.001) << std::endl;

    nn.train();


    return 0;
}
