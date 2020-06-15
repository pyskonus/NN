//
// Created by pyskonus on 6/13/2020.
//

#include "NN.h"


extern std::condition_variable condom;
extern std::mutex muteks;

void debug_in_file(double* array, size_t size) {
    std::ofstream ofs{"../data/dummy.dat"};
    for(size_t i = 0; i < size; ++i)
        ofs << array[i] << std::endl;
    ofs.close();
}

void sigmoid(double* array, size_t size) {
    for(size_t i = 0; i < size; ++i) {
        array[i] = 1.0/(1.0 + std::exp(-array[i]));
    }
}

void sigmoid_grad(double* array, size_t size) {
    for(size_t i = 0; i < size; ++i) {
        array[i] = 1.0/(1.0 + std::exp(-array[i])) * (1-1.0/(1.0 + std::exp(-array[i])));
    }
}

void diff(const double* a, const double* b, double* res, size_t size) {
    for(size_t i = 0; i < size; ++i) {
        res[i] = a[i] - b[i];
    }
}

void mult(const double* a, const double* b, double* res, size_t size) {
    for(size_t i = 0; i < size; ++i) {
        res[i] = a[i] * b[i];
    }
}

double* vec_vec_mat(const double* v1, const double* v2, size_t s1, size_t s2) {
    auto res = new double[s1*s2];

    for(size_t i = 0; i < s1; ++i) {
        for(size_t j = 0; j < s2; ++j) {
            res[i*s2+j] = v1[i]*v2[j];
        }
    }
    return res;
}

double* one_iteration(double* tr_example, const std::vector<Matrix>& thetas) {
    double* prev;
    double* res;

    res = new double[thetas[0].size.first+1];
    thetas[0].mat_vec(tr_example, res+1);
    sigmoid(res+1, thetas[0].size.first);
    res[0] = 1;
    prev = res;
    for(size_t i = 1; i < thetas.size()-1; ++i) {
        res = new double[thetas[i].size.first+1];
        thetas[i].mat_vec(prev, res+1);
        sigmoid(res+1, thetas[i].size.first);
        res[0] = 1;
        delete[] prev;
        prev = res;
    }
    res = new double[thetas[thetas.size()-1].size.first];
    thetas[thetas.size()-1].mat_vec(prev, res);
    sigmoid(res, thetas[thetas.size()-1].size.first);
    /*std::cout << res[0] << " " << res[1] << " " << res[2] << " " << res[3] << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));*/
    delete[] prev;
    return res;
}

bool is_zero(const double* array, size_t size) {
    bool res = true;
    for(size_t i = 0; i < size; ++i) {
        if(array[i]!=0)
            res=false;
    }
    return res;
}

void set_zero(double* array, size_t size) {
    for(size_t i = 0; i < size; ++i) {
        array[i] = 0;
    }
}

void distributor(std::deque<double*>& q, const std::string& tr_path, size_t len) {
    size_t counter = 0;
    std::ifstream tr_examples{tr_path};
    while(true) {
        std::unique_lock<std::mutex> locker(muteks);
        q.push_front(new double[len]);
        for(size_t i = 0; i < len; ++i) {
            tr_examples >> q.front()[i];
        }
        if(tr_examples.eof()) {
            set_zero(q.front(), len);
            counter++;
        }
        /*std::cout << q.front()[0] << " " << counter << " " << tr_examples.eof() << std::endl;*/
        if(counter%(std::thread::hardware_concurrency())==0 && counter!=0)
            break;

        locker.unlock();
        condom.notify_one();
        /*std::this_thread::sleep_for(std::chrono::milliseconds(1000));*/
    }
    tr_examples.close();

}

void listener(std::deque<double*>& q, std::vector<double*>& v, size_t len, const std::vector<Matrix>& thetas) {
    double* data;
    while(true) {
        std::unique_lock<std::mutex> locker(muteks);
        condom.wait(locker, [&q]{return !q.empty();});
        data=q.back();
        q.pop_back();
        /*std::this_thread::sleep_for(std::chrono::milliseconds(100));*/
        /*std::cout << data[0] << " " << data[1] << " " << data[2] << " " << data[3] << std::endl;*/
        /*std::cout << data[0] << std::endl;*/
        if(is_zero(data, len)) {
            /*std::cout << "zero at thread id " << std::this_thread::get_id() << std::endl;*/
            break;
        } else {
            v.emplace_back(one_iteration(data, thetas));
        }
        delete[] data;
        locker.unlock();
    }
}

std::vector<double*> propagate(const std::string& tr_path, const std::vector<Matrix>& thetas) {
    auto q = std::deque<double*>();
    std::vector<double*> v;

    auto threads = std::vector<std::thread>(std::thread::hardware_concurrency());
    threads[0] = std::thread{distributor, std::ref(q), std::ref(tr_path), thetas[0].size.second};
    for(size_t i = 1; i < std::thread::hardware_concurrency(); ++i) {
        threads[i] = std::thread{listener, std::ref(q), std::ref(v), thetas[0].size.second, std::ref(thetas)};
    }

    for(auto& t: threads)
        t.join();

    return v;
}

void grad_one(double* tr_example, double* res_example, const std::vector<Matrix>& thetas, std::vector<Matrix>& grads) {
    /// forward prop
    std::vector<double*> Zs;
    std::vector<double*> Ds;
    std::vector<double*> As;
    As.emplace_back(tr_example);
    for(const auto & theta : thetas) {
        Ds.emplace_back(new double[theta.size.first]);
    }
    /*for(size_t i = 0; i < thetas.size()-1; ++i) {
        Ds.emplace_back(new double[thetas[i+1].size.second]);
    }
    Ds.emplace_back(new double[thetas.back().size.first]);*/
    double* prev;
    double* res;

    res = new double[thetas[0].size.first+1];
    thetas[0].mat_vec(tr_example, res+1);
    Zs.emplace_back(new double[thetas[0].size.first+1]);
    memcpy(Zs[0]+1, res+1, thetas[0].size.first*sizeof(double));
    Zs[0][0] = 1;
    sigmoid(res+1, thetas[0].size.first);
    res[0] = 1;
    As.emplace_back(new double[thetas[0].size.first+1]);
    memcpy(As[1], res, (thetas[0].size.first+1)*sizeof(double));
    prev = res;

    for(size_t i = 1; i < thetas.size()-1; ++i) {
        res = new double[thetas[i].size.first + 1];
        thetas[i].mat_vec(prev, res + 1);
        Zs.emplace_back(new double[thetas[i].size.first+1]);
        memcpy(Zs[i], res, (thetas[i].size.first+1)*sizeof(double));
        Zs[i][0] = 1;
        sigmoid(res + 1, thetas[i].size.first);
        res[0] = 1;
        As.emplace_back(new double[thetas[i].size.first+1]);
        memcpy(As[i+1], res, (thetas[i].size.first+1)*sizeof(double));
        delete[] prev;
        prev = res;
    }
    res = new double[thetas[thetas.size()-1].size.first];
    thetas[thetas.size()-1].mat_vec(prev, res);
    Zs.emplace_back(new double[thetas[thetas.size()-1].size.first]);
    memcpy(Zs[Zs.size()-1], res, thetas[thetas.size()-1].size.first*sizeof(double));
    sigmoid(res, thetas[thetas.size()-1].size.first);
    As.emplace_back(new double[thetas[thetas.size()-1].size.first]);
    memcpy(As[As.size()-1], res, thetas[thetas.size()-1].size.first*sizeof(double));
    /// forward prop done

    diff(As[As.size()-1], res_example, Ds[Ds.size()-1], thetas.back().size.first);
//    debug_in_file(Ds.back(), 4);

    for(size_t i = thetas.size()-1; i >= 1; --i) {
        auto els = new double[thetas[i].size.first*thetas[i].size.second];
        auto transpose = thetas[i].transpose(els);
        transpose.mat_vec(Ds[i], Ds[i-1]);
        sigmoid_grad(Zs[i-1]+1, thetas[i-1].size.first);
        mult(Ds[i-1]+1, Zs[i-1]+1, Ds[i-1], thetas[i-1].size.first);
        delete[] els;
    }

    for(size_t i = 0; i < thetas.size(); ++i) {
        double* temp;
        temp = vec_vec_mat(Ds[i], As[i], thetas[i].size.first, thetas[i].size.second);
        for(size_t j = 0; j < thetas[i].size.first*thetas[i].size.second; ++j) {
            grads[i].elements[j] += temp[j];
        }
        delete[] temp;
    }

    /*DELETE*/
    for(auto addr: Ds) {
        delete[] addr;
    }
    for(auto addr: Zs) {
        delete[] addr;
    }
    for(auto addr: As) {
        if(addr!=tr_example)
            delete[] addr;
    }
}

double cost(const std::string& tr_path, const std::string& res_path, const std::vector<Matrix>& thetas) {
    auto result = propagate(tr_path, thetas);
    double cost_res = 0;
    size_t j = 0;
    auto* delta = new double[thetas.back().size.first];
    auto* expected = new double[thetas.back().size.first];

    std::ifstream res_file{res_path};
    for(auto & i : result) {
        for(j = 0; j < thetas.back().size.first; ++j)
            res_file >> expected[j];

        diff(i, expected, delta, thetas.back().size.first);
        cost_res += std::inner_product(delta, delta+thetas.back().size.first, delta, 0.0);
    }

    res_file.close();
    delete[] delta;
    delete[] expected;

    return cost_res;
}

void one_part(const std::vector<double*>& tr_examples, const std::vector<double*>& res_examples, size_t start,
        size_t finish, const std::vector<Matrix>& thetas, std::vector<Matrix>& grads) {
    /*std::lock_guard<std::mutex> lg(muteks);*/
    for(size_t i = start; i < finish; ++i) {
        grad_one(tr_examples[i], res_examples[i], thetas, grads);
    }
}

void backprop(const std::string& tr_path, const std::string& res_path, std::vector<Matrix>& thetas, double learning_rate, size_t iters) {
    /// read dataset
    std::vector<double*>tr_examples;
    std::vector<double*>res_examples;
    std::ifstream tr_file{tr_path};
    std::ifstream res_file{res_path};
    /// form the vectors consisting of rows of data
    while(true) {
        tr_examples.emplace_back(new double[thetas.front().size.second]);
        res_examples.emplace_back(new double[thetas.back().size.first]);
        for (size_t i = 0; i < thetas.front().size.second; ++i) {
            tr_file >> tr_examples[tr_examples.size()-1][i];
        }
        for (size_t i = 0; i < thetas.back().size.first; ++i) {
            res_file >> res_examples[res_examples.size()-1][i];
        }
        if(!res_file.good() || !tr_file.good()) {
            delete[] tr_examples[tr_examples.size()-1];
            tr_examples.pop_back();
            delete[] res_examples[tr_examples.size()-1];
            res_examples.pop_back();
            break;
        }
    }
    tr_file.close();
    res_file.close();

    /// set up delimiters
    auto* delims = new size_t[std::thread::hardware_concurrency()+1];
    for(size_t i = 0; i < std::thread::hardware_concurrency()+1; ++i) {
        delims[i] = i*tr_examples.size()/std::thread::hardware_concurrency();
    }

    for(size_t i = 0; i < iters; ++i) {
        std::vector<std::thread> threads(std::thread::hardware_concurrency());
        std::vector<std::vector<double*>> threads_grads_els;
        std::vector<std::vector<Matrix>> threads_grads;
        for(size_t j = 0; j < std::thread::hardware_concurrency(); ++j) {
            threads_grads_els.emplace_back(std::vector<double *>(thetas.size()));
            threads_grads.emplace_back(std::vector<Matrix>(thetas.size()));
            for (size_t k = 0; k < thetas.size(); ++k) {
                threads_grads_els[j][k] = new double[thetas[k].size.first * thetas[k].size.second];
                for (size_t m = 0; m < thetas[k].size.first * thetas[k].size.second; ++m)
                    threads_grads_els[j][k][m] = 0;
                threads_grads[j][k] = Matrix{thetas[k].size.first, thetas[k].size.second, threads_grads_els[j][k]};
            }
        }

        for(size_t j = 0; j < threads.size(); ++j) {
            threads[j] = std::thread(one_part, std::ref(tr_examples), std::ref(res_examples), delims[j], delims[j+1],
                    std::ref(thetas), std::ref(threads_grads[j]));
        }
        std::cout << "OK" << std::endl;
        for(auto& t: threads) {
            t.join();
        }
        for(const auto& thread$_grad: threads_grads) {
            for(size_t k = 0; k < thetas.size(); ++k) {
                for(size_t m = 0; m < thetas[k].size.first*thetas[k].size.second; ++m) {
                    thetas[k].elements[m] -= learning_rate*thread$_grad[k].elements[m];
                }
            }
        }
        /// update thetas
        for(auto& elss: threads_grads_els)
            for(auto& els: elss)
                delete[] els;
    }

    std::vector<std::string> filenames(thetas.size());
    for(size_t i = 0; i < thetas.size(); ++i) {
        filenames[i] = "../data/"+std::to_string(i+1)+".dat";
        thetas[i].print_file(filenames[i]);
    }
}
