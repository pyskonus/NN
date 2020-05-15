//
// Created by pyskonus on 5/14/2020.
//

#ifndef NN_NNETWORK_H
#define NN_NNETWORK_H

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <string>

class NNetwork {
public:
    explicit NNetwork(Eigen::MatrixXd& X_): X{X_} {training_examples_dim=X_.rows(); in_layer_dim=X_.cols();};
    explicit NNetwork(const std::vector<std::vector<double>>& X_);
    void set_thetas(std::vector<Eigen::MatrixXd>&);
    void set_y(Eigen::MatrixXd&);
    static Eigen::MatrixXd load_file(const std::string& X_path);
    Eigen::MatrixXd feed_forward();
    double cost();
    double cost_reg(double lambda);
    void train(const std::vector<std::pair<unsigned int, unsigned int>>& =
            std::vector<std::pair<unsigned int, unsigned int>>());
    void XX() {
        std::cout << X << std::endl;
    }
    void yy() {
        std::cout << y << std::endl;
    }

private:
    Eigen::MatrixXd X;  /// train input
    Eigen::MatrixXd y;  /// train output
    unsigned int training_examples_dim;
    unsigned int in_layer_dim, out_layer_dim{};
    std::vector<unsigned int> hidden_layers_dims{};     /// dimensions of hidden layers (without bias)
    std::vector<Eigen::MatrixXd> thetas;
//    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> forward_one_step();
    static Eigen::MatrixXd sigmoid(const Eigen::MatrixXd&);
};


#endif //NN_NNETWORK_H
