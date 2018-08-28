#ifndef UTILS_COMMON_H
#define UTILS_COMMON_H

#include <armadillo>
#include <cassert>
#include <string>

#define ASSERT_MAT_SAME_SIZE(mat1, mat12)  assert (mat1.n_rows == mat2.n_rows && mat1.n_cols == mat2.n_cols)

struct grads {
    std::vector<arma::mat> dW;
    std::vector<arma::rowvec> db;
};

struct cache {
    arma::mat X;
    std::vector<arma::mat> z;
    std::vector<arma::mat> a;
    arma::mat yc;
};

/*
 * Applies the sigmoid function to each element of the matrix
 * and returns a new matrix.
 */
void sigmoid(const arma::mat& mat, arma::mat& mat2);

/*
 * ReLU activation
 */
void relu(const arma::mat& mat, arma::mat& mat2);

/*
 * Applies the softmax to each rowvec of the matrix
 */
void softmax(const arma::mat& mat, arma::mat& mat2);

/*
 * Performs gradient check by comparing numerical and analytical gradients.
 */
bool gradcheck(struct grads& grads1, struct grads& grads2);

/*
 * Compares the two label vectors to compute precision.
 */
double precision(arma::vec vec1, arma::vec vec2);

/*
 * Converts label vector into a matrix of one-hot label vectors
 * @params label : label vector
 * @params C : Number of classes
 * @params [out] y : The y matrix.
 */
void label_to_y(arma::vec label, int C, arma::mat& y);

void save_label(std::string filename, arma::vec& label);

#endif
