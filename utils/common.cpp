#include "common.h"

#include <iostream>

#define MAX_REL_ERROR_THRESHOLD 1000

void sigmoid(const arma::mat& mat, arma::mat& mat2) {
    mat2.set_size(mat.n_rows, mat.n_cols);
    ASSERT_MAT_SAME_SIZE(mat, mat2);
    mat2 = 1 / (1 + arma::exp(-mat));
}

void softmax(const arma::mat& mat, arma::mat& mat2) {
    mat2.set_size(mat.n_rows, mat.n_cols);
    arma::mat exp_mat = arma::exp(mat);
    arma::mat sum_exp_mat = arma::sum(exp_mat, 1);
    mat2 = exp_mat / repmat(sum_exp_mat, 1, mat.n_cols);
}

/*
 * Returns the relative error between two matrices.
 */
double rel_error(arma::mat& mat1, arma::mat& mat2) {
    ASSERT_MAT_SAME_SIZE(mat1, mat2);
    arma::mat threshold = arma::mat(mat1.n_rows, mat1.n_cols, arma::fill::ones);
    return arma::max(arma::max(arma::abs(mat1 - mat2) / arma::max(threshold,
                               arma::max(arma::abs(mat1),  arma::abs(mat2)))));
}

/*
 * Performs gradient check
 */
bool gradcheck(struct grads& grads1, struct grads& grads2) {
    assert(grads1.dW.size() == grads2.dW.size());
    assert(grads1.db.size() == grads2.db.size());

    for(int i = grads1.dW.size() - 1; i >=0; --i) {
        double error = rel_error(grads1.dW[i], grads2.dW[i]);
        std::cout << "dW[" << i << "] rel error: " << error << "\n";

        if(error > MAX_REL_ERROR_THRESHOLD) {
            return false;
        }
    }

    for(int i = grads1.db.size() - 1; i >=0; --i) {
        double error = rel_error(grads1.db[i], grads2.db[i]);
        std::cout << "db[" << i << "] rel error: " << error << "\n";

        if(error > MAX_REL_ERROR_THRESHOLD) {
            return false;
        }
    }

    return true;
}

/*
 * Converts label vector into a matrix of one-hot label vectors.
 */
void label_to_y(arma::vec label, int C, arma::mat& y) {
    y.set_size(label.size(), C);
    y.fill(0);

    for(int i = 0; i < label.size(); ++i) {
        y(i, label(i)) = 1;
    }
}

/*
 * ReLu activation.
 */
void relu(const arma::mat& mat, arma::mat& mat2) {
    mat2 = mat;
    mat2.elem(arma::find(mat < 0)).zeros();
}

double precision(arma::vec vec1, arma::vec vec2) {
    return arma::accu(vec1 == vec2) / (float) vec1.size();
}

void save_label(std::string filename, arma::vec& label) {
    std::ofstream file(filename);

    if(file.is_open()) {
        for(int i = 0; i < label.size(); ++i) {
            file << label(i);
        }

        file.close();
    } else {
        std::cerr << "Save label to file " << filename << " failed!" << std::endl;
    }
}
