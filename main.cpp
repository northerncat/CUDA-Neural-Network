#include <iostream>
#include <cassert>
#include <mpi.h>
#include <armadillo>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <unistd.h>

#include "utils/mnist.h"
#include "two_layer_net.h"
#include "utils/tests.h"
#include "utils/common.h"
#include "gpu_func.h"

#define FILE_TRAIN_IMAGES "data/train-images-idx3-ubyte"
#define FILE_TRAIN_LABELS "data/train-labels-idx1-ubyte"
#define FILE_TEST_IMAGES "data/t10k-images-idx3-ubyte"
#define FILE_TEST_OUTPUT "Outputs/Pred_testset.txt"
#define NUM_TRAIN 60000
#define IMAGE_SIZE 784  // 28 x 28
#define NUM_CLASSES 10
#define NUM_TEST 10000

#define MPI_SAFE_CALL( call ) do {                               \
    int err = call;                                              \
    if (err != MPI_SUCCESS) {                                    \
        fprintf(stderr, "MPI error %d in file '%s' at line %i",  \
               err, __FILE__, __LINE__);                         \
        exit(1);                                                 \
    } } while(0)

int main(int argc, char* argv[]) {
    // Initialize MPI
    int num_procs = 0, rank = 0;
    MPI_Init(&argc, &argv);
    MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
    MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    // Assign a GPU device to each MPI proc
    int nDevices;
    cudaGetDeviceCount(&nDevices);

    if(nDevices < num_procs) {
        std::cerr << "Please allocate at least as many GPUs as\
		 the number of MPI procs." << std::endl;
    }

    checkCudaErrors(cudaSetDevice(rank));

    if(rank == 0) {
        std::cout << "Number of MPI processes = " << num_procs << std::endl;
        std::cout << "Number of CUDA devices = " << nDevices << std::endl;
    }

    // Read in command line arguments
    std::vector<int> H(3);
    double reg = 1e-4;
    double learning_rate = 0.001;
    int num_epochs = 20;
    int batch_size = 800;
    int num_neuron = 1000;
    int run_seq = 0;
    int debug = 0;
    int grade = 0;
    int print_every = 0;

    int option = 0;

    while((option = getopt(argc, argv, "n:r:l:e:b:g:p:sd")) != -1) {
        switch(option) {
            case 'n':
                num_neuron = atoi(optarg);
                break;

            case 'r':
                reg = atof(optarg);
                break;

            case 'l':
                learning_rate = atof(optarg);
                break;

            case 'e':
                num_epochs = atoi(optarg);
                break;

            case 'b':
                batch_size = atoi(optarg);
                break;

            case 'g':
                grade = atoi(optarg);
                break;

            case 'p':
                print_every = atoi(optarg);
                break;

            case 's':
                run_seq = 1;
                break;

            case 'd':
                debug = 1;
                break;
        }
    }

    /* This option is going to be used to test correctness.
       DO NOT change the following parameters */
    switch(grade) {
        case 0:  // No grading
            break;

        case 1:  // Low lr, high iters
            reg = 1e-4;
            learning_rate = 0.001;
            num_epochs = 40;
            batch_size = 800;
            num_neuron = 100;
            run_seq = 1;
            debug = 1;
            print_every = 0;
            break;

        case 2:  // Medium lr, medium iters
            reg = 1e-4;
            learning_rate = 0.01;
            num_epochs = 10;
            batch_size = 800;
            num_neuron = 100;
            run_seq = 1;
            debug = 1;
            print_every = 0;
            break;

        case 3:  // High lr, very few iters
            reg = 1e-4;
            learning_rate = 0.025;
            num_epochs = 1;
            batch_size = 800;
            num_neuron = 100;
            run_seq = 1;
            debug = 1;
            print_every = 1;
            break;

        case 4:
            break;
    }

    if(grade == 4) {
        if(rank == 0) {
            BenchmarkGEMM();
        }

        MPI_Finalize();
        return 0;
    }

    H[0] = IMAGE_SIZE;
    H[1] = num_neuron;
    H[2] = NUM_CLASSES;

    arma::mat x_train, y_train, label_train, x_dev, y_dev, label_dev, x_test;
    TwoLayerNet nn(H);

    if(rank == 0) {
        std::cout << "num_neuron=" << num_neuron << ", reg=" << reg <<
                  ", learning_rate=" << learning_rate
                  << ", num_epochs=" << num_epochs << ", batch_size=" << batch_size << std::endl;
        // Read MNIST images into Armadillo mat vector
        arma::mat x(NUM_TRAIN, IMAGE_SIZE);
        // label contains the prediction for each
        arma::colvec label = arma::zeros<arma::colvec>(NUM_TRAIN);
        // y is the matrix of one-hot label vectors where only y[c] = 1,
        // where c is the right class.
        arma::mat y = arma::zeros<arma::mat>(NUM_TRAIN, NUM_CLASSES);

        std::cout << "Loading training data..." << std::endl;
        read_mnist(FILE_TRAIN_IMAGES, x);
        read_mnist_label(FILE_TRAIN_LABELS, label);
        label_to_y(label, NUM_CLASSES, y);

        /* Print stats of training data */
        std::cout << "Training data stats..." << std::endl;
        std::cout << "Size of x_train, N =  " << x.n_rows << std::endl;
        std::cout << "Size of label_train = " << label.size() << std::endl;

        assert(x.n_rows == NUM_TRAIN && x.n_cols == IMAGE_SIZE);
        assert(label.size() == NUM_TRAIN);

        /* Split into train set and dev set, you should use train set to train your
           neural network and dev set to evaluate its precision */
        int dev_size = (int)(0.1 * NUM_TRAIN);
        x_train = x.rows(0, NUM_TRAIN-dev_size-1);
        y_train = y.rows(0, NUM_TRAIN-dev_size-1);
        label_train = label.rows(0, NUM_TRAIN-dev_size-1);

        x_dev = x.rows(NUM_TRAIN-dev_size, NUM_TRAIN - 1);
        y_dev = y.rows(NUM_TRAIN-dev_size, NUM_TRAIN - 1);
        label_dev = label.rows(NUM_TRAIN-dev_size, NUM_TRAIN - 1);

        /* Load the test data, we will compare the prediction of your trained neural
           network with test data label to evaluate its precision */
        x_test = arma::zeros(NUM_TEST, IMAGE_SIZE);
        read_mnist(FILE_TEST_IMAGES, x_test);
    }

    /* Run the sequential code if the serial flag is set */
    TwoLayerNet seq_nn(H);

    if((rank == 0) && (run_seq)) {
        std::cout << "Start Sequential Training" << std::endl;

        double start = MPI_Wtime();
        train(seq_nn, x_train, y_train, learning_rate, reg, num_epochs, batch_size,
              false, print_every, debug);
        double end = MPI_Wtime();

        std::cout << "Time for Sequential Training: " << end - start << " seconds" <<
                  std::endl;

        arma::vec label_dev_pred;
        predict(seq_nn, x_dev, label_dev_pred);
        double prec = precision(label_dev_pred, label_dev);
        std::cout << "Precision on validation set for sequential training = " << prec <<
                  std::endl;
    }

    /* Train the Neural Network in Parallel*/
    if(rank == 0) {
        std::cout << std::endl << "Start Parallel Training" << std::endl;
    }

    double start = MPI_Wtime();

    /* ---- Parallel Training ---- */
    parallel_train(nn, x_train, y_train, learning_rate, reg, num_epochs, batch_size,
                   false, print_every, debug);

    double end = MPI_Wtime();

    if(rank == 0) {
        std::cout << "Time for Parallel Training: " << end - start << " seconds" <<
                  std::endl;
    }

    /* Note: Make sure after parallel training, rank 0's neural network is up to date */

    /* Do predictions for the parallel NN */
    if(rank == 0) {
        arma::vec label_dev_pred;
        predict(nn, x_dev, label_dev_pred);
        double prec = precision(label_dev_pred, label_dev);
        std::cout << "Precision on validation set for parallel training = " << prec <<
                  std::endl;
        arma::vec label_test_pred;
        predict(nn, x_test, label_test_pred);
        save_label(FILE_TEST_OUTPUT, label_test_pred);
    }

    /* If grading mode is on, compare CPU and GPU results and check for correctness */
    if((grade || debug) && rank == 0) {
        std::cout << std::endl << "Grading mode on. Checking for correctness" <<
                  std::endl;
        checkNNErrors(seq_nn, nn, "Outputs/NNErrors.txt");
    }

    MPI_Finalize();
    return 0;
}
