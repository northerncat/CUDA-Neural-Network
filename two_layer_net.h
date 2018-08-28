#ifndef UTILS_TWO_LAYER_NET_H_
#define UTILS_TWO_LAYER_NET_H_

#include <armadillo>
#include <cmath>
#include <iostream>

class TwoLayerNet
{
public:
	const int num_layers = 2;
	// H[i] is the number of neurons in layer i (where i=0 implies input layer)
	std::vector<int> H;
	// Weights of the neural network
    // W[i] are the weights of the i^th layer
	std::vector<arma::mat> W;
	// Biases of the neural network
    // b[i] is the row vector biases of the i^th layer
    std::vector<arma::rowvec> b;

    TwoLayerNet (std::vector<int> _H) {
    	W.resize (num_layers);
    	b.resize (num_layers);
    	H = _H;

    	for (int i = 0; i < num_layers; i++) {
            arma::arma_rng::set_seed(arma::arma_rng::seed_type(i));
    		W[i] = 0.01 * arma::randn (H[i+1], H[i]);
    		b[i].zeros (H[i+1]);
    	}
    }
};

void feedforward (TwoLayerNet &nn, const arma::mat& X, struct cache& bpcache);
double loss (TwoLayerNet &nn, const arma::mat& yc, const arma::mat& y, double reg);
void backprop (TwoLayerNet &nn, const arma::mat& y, double reg, const struct cache& bpcache, struct grads& bpgrads);
void numgrad (TwoLayerNet &nn, const arma::mat& X, const arma::mat& y, double reg, struct grads& numgrads);
void train (TwoLayerNet &nn, const arma::mat& X, const arma::mat& y, double learning_rate, double reg = 0.0, const int epochs = 15, const int batch_size = 800, bool grad_check = false, int print_every = -1, int debug = 0);
void predict (TwoLayerNet &nn, const arma::mat& X, arma::mat& label);

void parallel_train (TwoLayerNet &nn, const arma::mat& X, const arma::mat& y, double learning_rate, double reg = 0.0, const int epochs = 15, const int batch_size = 800, bool grad_check = false, int print_every = -1, int debug = 0);

#endif