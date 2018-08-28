#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include <vector>

struct event_pair
{
  cudaEvent_t start;
  cudaEvent_t end;
};

inline void check_launch(const char * kernel_name)
{
  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess)
  {
    std::cerr << "error in " << kernel_name << " kernel" << std::endl;
    std::cerr << "error was: " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
}

inline void start_timer(event_pair * p)
{
  cudaEventCreate(&p->start);
  cudaEventCreate(&p->end);
  cudaEventRecord(p->start, 0);
}


inline double stop_timer(event_pair * p)
{
  cudaEventRecord(p->end, 0);
  cudaEventSynchronize(p->end);
  
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, p->start, p->end);
  cudaEventDestroy(p->start);
  cudaEventDestroy(p->end);
  return elapsed_time;
}

int useless_gpu_add_one (int t);

int myGEMM(const double* A, const double* B, double* C,
           double* alpha, double* beta, int M, int N, int K);

/**
 * Function: myAllocGEMM
 * ---------------------
 * This function computes GEMM the same way as myGEMM, but not in place. The results
 * would be stored in the pointer D, which is conditionally allocated based on the
 * allocate variable.
 */
int myAllocGEMM(const double* A, const double* B, const double* C, double*& D,
                double alpha, double beta, int M, int N, int K, bool allocate);

/**
 * Function: computeZ
 * ------------------
 * This function computes GEMM the same way as myGEMM, but not in place and the second
 * matrix b is now a vector instead of a matrix. The function is required because in
 * computing z of each layer, the bias is a vector and doing so can avoid performing
 * operations similar to repmat the bias by the number of samples. The results would be
 * stored in the pointer z, which is allocated within the function.
 */
void computeZ(const double* a, const double* W, const double* b, double*& z, int N, int M, int H);

/**
 * Function: mySigmoid
 * -------------------
 * This function computes the sigmoid of a matrix with the given dimension in-place.
 */
int mySigmoid(double* mat, int M, int N);

/**
 * Function: mySoftmax
 * -------------------
 * This function computes the softmax of a matrix with the given dimension in-place,
 * assuming that each row of the matrix is the outputs from one sample.
 */
int mySoftmax(double* mat, int N, int C);

/**
 * Function: elemMults
 * -------------------
 * This function computes the elementwise multiplication as used when computing the
 * derivative of CE with z1, aka dCE/dz1 = dCE/da1 o a1 o (1-a1).
 */
void elemMults(const double* A, const double* B, double*& C, int M, int N);

/**
 * Function: columnSums
 * --------------------
 * This function computes the column sum of the matrix and stores the results in sums,
 * assuming that the sums pointer has already been properly allocated.
 */
void columnSums(const double* mat, double* sums, int M, int N);

/**
 * Function: matAdd, vecAdd
 * ------------------------
 * These functions compute the sum of two matrices or two vectors with the given coefs
 * and stores the results in the first matrix or vector
 */
void matAdd(double* X, double* Y, double a, double b, int M, int N);
void vecAdd(double* X, double* Y, double a, double b, int M);

/**
 * Function: transpose
 * -------------------
 * This function computes the transpose of the matrix A with dimension MxN and stores
 * the results AT, which is allocated within the function.
 */
void transpose(double* A, double*& AT, int M, int N);

/**
 * Function: freePtrs
 * ------------------
 * This function frees all the device pointers stored in a vector of pointers.
 */
void freePtrs(std::vector<double*> ptrs);

/**
 * Class: CachePtr
 * ---------------
 * The CachePtr class is the pointer version of struct cache, except that X is stored
 * as transposed, y is stored as the difference between prediction and expected, and only
 * a1 is stored but not z1, a2 and z2 since they're not used in backprop.
 */
class CachePtr {
public:
    double* X;
    double* yDiff;
    double* a1;
    int N;
    std::vector<int> H;
    bool isDevice;

    CachePtr (std::vector<int> _H, int _N, bool device) : N(_N), H(_H), isDevice(device) {}

    ~CachePtr() {
        checkCudaErrors ( cudaFree(yDiff) );
        checkCudaErrors ( cudaFree(a1) );
    }
};

/**
 * Class: GradPtr
 * --------------
 * The GradPtr class is the pointer version of struct grads, except that the pointers
 * can be device pointers or host pointers, and is controlled by a boolean isDevice.
 */
class GradPtr {
public:
    std::vector<double*> dW;
    std::vector<double*> db;
    std::vector<int> H;
    int num_layers;
    bool isDevice;

    GradPtr (std::vector<int> _H, bool device) : H(_H), isDevice(device), num_layers(2) {
        dW.resize(num_layers);
        db.resize(num_layers);

        if (isDevice) {
            for (int i = 0; i < num_layers; ++i) {
                checkCudaErrors( cudaMalloc((void**) &dW[i], H[i] * H[i+1] * sizeof(double)) );
                checkCudaErrors( cudaMalloc((void**) &db[i], H[i+1] * sizeof(double)) );
            }
        } else {
            for (int i = 0; i < num_layers; ++i) {
                dW[i] = (double*) malloc(H[i] * H[i+1] * sizeof(double));
                db[i] = (double*) malloc(H[i+1] * sizeof(double));
            }
        }
    }

    ~GradPtr() {
        if (isDevice) {
            for (int i = 0; i < num_layers; ++i) {
                checkCudaErrors( cudaFree(dW[i]) );
                checkCudaErrors( cudaFree(db[i]) );
            }
        } else {
            for (int i = 0; i < num_layers; ++i) {
                free(dW[i]);
                free(db[i]);
            }
        }
    }
};

/**
 * Class: DataPointers
 * -------------------
 * The DataPointers class stores pointers to the transpose of X and y, and the
 * pointers can be on the device or host.
 */
class DataPointers {
public:
    // transposes of the training data. Each columns is one sample, making
    // it easier to access certain samples
    double* XT;
    double* yT;

    // XT should be a column major storage matrix with dim NxH0
    int N;
    int H0;
    // yT should be of dim NxC
    int C;

    bool isDevice;

    DataPointers (int _N, int _H0, int _C, bool device) : N(_N), H0(_H0), C(_C), isDevice(device) {
        if (isDevice) {
            checkCudaErrors( cudaMalloc((void**)&XT, N * H0 * sizeof(double)) );
            checkCudaErrors( cudaMalloc((void**)&yT, N * C * sizeof(double)) );
        } else {
            XT = (double*) malloc(N * H0 * sizeof(double));
            yT = (double*) malloc(N * C * sizeof(double));
        }
    }

    ~DataPointers() {
        if (isDevice) {
            checkCudaErrors ( cudaFree(XT) );
            checkCudaErrors ( cudaFree(yT) );
        } else {
            free(XT);
            free(yT);
        }
    }
};

/**
 * Class: NNPointers
 * -----------------
 * The NNPointers class stores pointers to the transpose of Ws, and bs, and the
 * pointers can be on the device or host.
 */
class NNPointers {
public:
    // Weights of the neural network. W[i] are the weights of the i^th layer, 
    // here we stored it as the transpose of the orginal NN class
    std::vector<double*> W;
    // Biases of the neural network. b[i] is the row vector biases of the i^th layer
    std::vector<double*> b;
    // H[i] is the number of neurons in layer i (where i=0 implies input layer)
    std::vector<int> H;
    int num_layers;
    bool isDevice;

    NNPointers (std::vector<int> _H, bool device) : H(_H), isDevice(device), num_layers(2) {
        W.resize(num_layers);
        b.resize(num_layers);

        if (isDevice) {
            for (int i = 0; i < num_layers; ++i) {
                checkCudaErrors( cudaMalloc((void**) &W[i], H[i] * H[i+1] * sizeof(double)) );
                checkCudaErrors( cudaMalloc((void**) &b[i], H[i+1] * sizeof(double)) );
            }
        } else {
            for (int i = 0; i < num_layers; ++i) {
                W[i] = (double*) malloc(H[i] * H[i+1] * sizeof(double));
                b[i] = (double*) malloc(H[i+1] * sizeof(double));
            }
        }
    }

    ~NNPointers() {
        if (isDevice) {
            for (int i = 0; i < num_layers; ++i) {
                checkCudaErrors( cudaFree(W[i]) );
                checkCudaErrors( cudaFree(b[i]) );
            }
        } else {
            for (int i = 0; i < num_layers; ++i) {
                free(W[i]);
                free(b[i]);
            }
        }
    }
};

int copyToDevice(const DataPointers& hostPtrs, DataPointers& devicePtrs);
int copyToDevice(const NNPointers& hostNN, NNPointers& deviceNN);
int copyToDevice(const GradPtr& hostGrad, GradPtr& deviceGrad);
void copyToDevice(const double* hostPtr, double* devicePtr, unsigned int size);

int copyToHost(const DataPointers& devicePtrs, DataPointers& hostPtrs);
int copyToHost(const NNPointers& deviceNN, NNPointers& hostNN);
int copyToHost(const GradPtr& deviceGrad, GradPtr& hostGrad);
int copyToHost(const CachePtr& deviceCache, CachePtr& hostCache);
void copyToHost(const double* devicePtr, double* hostPtr, unsigned int size);

#endif
