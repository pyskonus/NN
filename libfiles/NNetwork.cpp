//
// Created by pyskonus on 5/14/2020.
//

#include "NNetwork.h"

NNetwork::NNetwork(const std::vector<std::vector<double>>& X_) {

    training_examples_dim = X_.size();
    in_layer_dim = X_[0].size();
    X = Eigen::MatrixXd(training_examples_dim, in_layer_dim);   /// set the size of X

    unsigned int X_row_length = X_[0].size();
    for (int i = 0; i < training_examples_dim; ++i) {
        for (int j = 0; j < in_layer_dim; ++j)
            X(i, j) = X_[i][j];
        if(X_[i].size() != X_row_length)
            std::cerr << "Warning: not all rows of the first matrix have same length" << std::endl;
    }
}

void NNetwork::set_thetas(std::vector<Eigen::MatrixXd>& thetas_) {
    if(thetas_.empty())
        throw std::invalid_argument("argument cannot be empty");
    /// go through the vector and check if the dimensions match
    if (thetas_[0].cols() != in_layer_dim + 1)
        throw std::invalid_argument("Invalid number of columns in theta 0. Must be " + std::to_string(in_layer_dim+1));
    /// Each next theta matrix must have amount of columns same as number of rows
    /// for the previous matrix but + 1 due to the bias unit.
    /// Once users cope with error above they will hopefully not encounter these:
    for(int i = 1; i < thetas_.size(); ++i) {
        if(thetas_[i].cols() != (thetas_[i-1].rows() +1))
            throw std::invalid_argument("Invalid number of columns in thetas[" + std::to_string(i) +
            "]. Must be " + std::to_string(thetas_[i-1].rows()+1));
    }

    thetas = thetas_;
}

void NNetwork::set_y(Eigen::MatrixXd& y_) {
    if(y_.size() == 0)
        throw std::invalid_argument("parameter cannot be empty");

    y = y_;
}

Eigen::MatrixXd NNetwork::feed_forward() {
    if(thetas.empty())
        throw std::runtime_error("train or set explicitly the network parameters first (thetas)");

    auto Zs = std::vector<Eigen::MatrixXd>();
    auto a = X;
    /// add a column of ones - bias unit:
    for(const auto& theta: thetas) {
        a.conservativeResize(a.rows(), a.cols()+1);
        a.col(a.cols()-1) = Eigen::ArrayXd::LinSpaced(a.rows(),1,1);
        a = sigmoid(a * theta.transpose());
    }
    return a;
}

double NNetwork::cost() {
    /// no regularization
    if(thetas.empty())
        throw std::runtime_error("train or set explicitly the network parameters first (thetas)");
    if(y.size() == 0)
        throw std::runtime_error("load result samples to estimate the cost");
    if(y.cols() != thetas[thetas.size()-1].rows())
        throw std::runtime_error("output dimensions mismatch");

    auto obtained = this->feed_forward();

    return -1.0/training_examples_dim * (y.array() * log(obtained.array()).array() +
    (1-y.array()) * log(1-obtained.array()).array()).sum();
}

double NNetwork::cost_reg(double lambda) {
    /// with regularization
    double result = this->cost();
    double reg_term = 0;
    for(const auto& m: thetas)
        for(int i = 0; i < m.rows(); ++i)
            for(int j = 0; j < m.cols(); ++j)
                reg_term += m(i, j)*m(i, j);
    return this->cost() + reg_term*lambda/2/training_examples_dim;
}

void NNetwork::train(const std::vector<std::pair<unsigned int, unsigned int>>&) {
    /// will be implemented once I get the minimization library
}

Eigen::MatrixXd NNetwork::load_file(const std::string& path) {
    std::ifstream infile{path};

    unsigned int rows, cols;
    infile >> rows >> cols;

    if(!infile.good())  /// could not even read the dimension
        throw std::invalid_argument("File is not complete. Must contain matrix dimensions followed by the matrix itself.");

    auto result = Eigen::MatrixXd(rows, cols);   /// set the size of X
    for(int i = 0; i < rows; ++i)
        for(int j = 0; j < cols; ++j)
            infile >> result(i, j);
    if(infile.eof())
        std::cerr << path << " might be incomplete" << std::endl;
    /// If there is any whitespace character after the matrix in infile and its matrix
    /// is complete, that is, contains rows * cols elements,
    /// no output to cerr will happen. Otherwise, if the matrix is not complete,
    /// the program will fill the rest of result matrix with zeroes and resume.
    infile.close();
    return result;
}

Eigen::MatrixXd NNetwork::sigmoid(const Eigen::MatrixXd& arg) {
    Eigen::MatrixXd result = Eigen::MatrixXd(arg.rows(), arg.cols());
    for(int i = 0; i < arg.rows(); ++i)
        for(int j = 0; j < arg.cols(); ++j)
            result(i, j) = 1/(1 + std::exp(-arg(i, j)));

    return result;
}