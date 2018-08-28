#ifndef _UTILS_MNIST_H_
#define _UTILS_MNIST_H_

#include <cstring>
#include <armadillo>

int reverse_int(int i);
void read_mnist(std::string filename, arma::mat& mat);
void read_mnist_label(std::string filename, arma::colvec& vec);

#endif