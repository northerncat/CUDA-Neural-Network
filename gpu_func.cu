#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"

__global__
void device_add_one (int* d_result, int t)
{
	*d_result = t + 1;
}

/*
Just a dummy function the can be used to warm up GPU
*/
int useless_gpu_add_one (int t)
{
	int result;
	int *d_result;

	checkCudaErrors (cudaMalloc((void **)&d_result, 1 * sizeof (int)));

	event_pair timer;
	start_timer (&timer);
	device_add_one<<<1,1>>>(d_result, t);
	check_launch ("device_add_one");
	double time = stop_timer (&timer);

	std::cout << "device_add_one took: " << time << " seconds" << std::endl;

	checkCudaErrors (cudaMemcpy(&result, d_result, 1 * sizeof (int), cudaMemcpyDeviceToHost));
	return result;
}


__global__
void simpleGEMM(const double* A, const double* B, double* C,
                double alpha, double beta, int M, int N, int K) {
    const int col = threadIdx.x + blockDim.x * blockIdx.x;
    const int row = threadIdx.y + blockDim.y * blockIdx.y;

    if (col >= N || row >= M) return;

    double sum = 0.0;
    for (int i = 0; i < K; ++i) {
        sum += A[i * M + row] * B[col * K + i];
    }
    C[col * M + row] = alpha * sum + C[col * M + row] * beta;
}


__global__
void optimizedGEMM(const double* A, const double* B, double* C,
                   double alpha, double beta, int M, int N, int K) {
    const int tr = threadIdx.x + blockDim.x * threadIdx.y;
    const int gStartR = blockIdx.x * 64;
    const int gStartC = blockIdx.y * 16;

    const int gr = gStartR + tr;

    if (gStartR >= M || gStartC >= N) return;

    __shared__ double bSM[2][33];

    double cols[16];

    for (int c = 0; c < 16; ++c) {
        cols[c] = 0.0;
    }

    int nIt = (K+3) / 4;
    for (int it = 0; it < nIt; ++it) {

        int smR = tr % 4;
        int smC = tr / 4;
        int br = it * 4 + smR;
        int bc = gStartC + smC;

        if (br < K && bc < N) {
            bSM[smR / 2][smC + (smR % 2) * 16] = B[bc * K + br];
        }

        __syncthreads();

        double aCache[4];

        if (gr < M) {
            for (int i = 0; i < 4; ++i) {
                if (it * 4 + i >= K) {
                    aCache[i] = 0.0;
                } else {
                    aCache[i] = A[(it * 4 + i) * M + gr];
                }
            }
        }

        for (int c = 0; c < 16; ++c) {
            for (int i = 0; i < 4; ++i) {
                double a = aCache[i];
                double b = bSM[i / 2][c + (i % 2) * 16];
                cols[c] += a * b;
            }
        }

        __syncthreads();

    }

    if (gr >= M) return;

    for (int c = 0; c < 16; ++c) {
        if (gStartC + c >= N) break;
        C[(gStartC + c) * M + gr] = cols[c] * alpha + C[(gStartC + c) * M + gr] * beta;
    }

}

/* 
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C 
*/
int myGEMM(const double* A, const double* B, double* C,
           double* alpha, double* beta, int M, int N, int K) {
    /* TODO: Write an efficient GEMM implementation on GPU */
    dim3 threadDims(16, 4);
    dim3 blockDims((M + 63) / 64, (N + 15) / 16);
    optimizedGEMM<<< blockDims, threadDims>>> (A, B, C, *alpha, *beta, M, N, K);

    return 0;
}

__global__
void allocGEMM(const double* A, const double* B, const double* C, double* D,
               double alpha, double beta, int M, int N, int K) {

    const int tr = threadIdx.x + blockDim.x * threadIdx.y;
    const int gStartR = blockIdx.x * 64;
    const int gStartC = blockIdx.y * 16;

    const int gr = gStartR + tr;

    if (gStartR >= M || gStartC >= N) return;

    __shared__ double bSM[2][33];

    double cols[16];

    for (int c = 0; c < 16; ++c) {
        cols[c] = 0.0;
    }

    int nIt = (K+3) / 4;
    for (int it = 0; it < nIt; ++it) {

        int smR = tr % 4;
        int smC = tr / 4;
        int br = it * 4 + smR;
        int bc = gStartC + smC;

        if (br < K && bc < N) {
            bSM[smR / 2][smC + (smR % 2) * 16] = B[bc * K + br];
        }

        __syncthreads();

        double aCache[4];

        if (gr < M) {
            for (int i = 0; i < 4; ++i) {
                aCache[i] = 0.0;
                if (it * 4 + i < K) {
                    aCache[i] = A[(it * 4 + i) * M + gr];
                }
            }

            for (int c = 0; c < 16; ++c) {
                for (int i = 0; i < 4; ++i) {
                    double a = aCache[i];
                    double b = bSM[i / 2][c + (i % 2) * 16];
                    cols[c] += a * b;
                }
            }
        }

        __syncthreads();

    }

    if (gr >= M) return;

    for (int c = 0; c < 16; ++c) {
        if (gStartC + c >= N) break;
        D[(gStartC + c) * M + gr] = cols[c] * alpha + C[(gStartC + c) * M + gr] * beta;
    }

}

/* 
Routine to allocate for D and perform a GEMM operation, i.e., D := alpha*A*B + beta*C 
*/
int myAllocGEMM(const double* A, const double* B, const double* C, double*& D,
                double alpha, double beta, int M, int N, int K, bool allocate) {
    /* TODO: Write an efficient GEMM implementation on GPU */
    if (allocate) {
        checkCudaErrors( cudaMalloc((void**) &D, M * N * sizeof(double)) );
    }

    // launch kernel
    dim3 threadDims(16, 4);
    dim3 blockDims((M + 63) / 64, (N + 15) / 16);
    allocGEMM<<<blockDims, threadDims>>>(A, B, C, D, alpha, beta, M, N, K);

    return 0;
}

/*************************************************************************************/
/***************************** GEMM for Vector b Kernel ******************************/
/*************************************************************************************/

// NxH = NxM x MxH + N x H
__global__
void gpuVecGEMM(const double* A, const double* B, const double* C, double* D, int M, int K, int N) {

    const int tr = threadIdx.x + blockDim.x * threadIdx.y;
    const int gStartR = blockIdx.x * 64;
    const int gStartC = blockIdx.y * 16;

    const int gr = gStartR + tr;

    if (gStartR >= M || gStartC >= N) return;

    __shared__ double bSM[2][33];

    double cols[16];

    for (int c = 0; c < 16; ++c) {
        cols[c] = 0.0;
    }

    int nIt = (K+3) / 4;
    for (int it = 0; it < nIt; ++it) {

        int smR = tr % 4;
        int smC = tr / 4;
        int br = it * 4 + smR;
        int bc = gStartC + smC;

        if (br < K && bc < N) {
            bSM[smR / 2][smC + (smR % 2) * 16] = B[bc * K + br];
        }

        __syncthreads();

        double aCache[4];

        if (gr < M) {
            for (int i = 0; i < 4; ++i) {
                aCache[i] = 0.0;
                if (it * 4 + i < K) {
                    aCache[i] = A[(it * 4 + i) * M + gr];
                }
            }

            for (int c = 0; c < 16; ++c) {
                for (int i = 0; i < 4; ++i) {
                    double a = aCache[i];
                    double b = bSM[i / 2][c + (i % 2) * 16];
                    cols[c] += a * b;
                }
            }
        }

        __syncthreads();

    }

    if (gr >= M) return;

    for (int c = 0; c < 16; ++c) {
        if (gStartC + c >= N) break;
        D[(gStartC + c) * M + gr] = cols[c] + C[gStartC + c];
    }

}

// compute z(NxH) as z = aW + repmat(b,N,1) for matrices a (NxM), W(MxH),
// and vector b of dim H. All matrices are column major
void computeZ(const double* a, const double* W, const double* b, double*& z, int N, int M, int H) {
    checkCudaErrors( cudaMalloc((void**) &z, N * H * sizeof(double)) );
    dim3 threadDims(16, 4);
    dim3 blockDims((N + 63) / 64, (H + 15) / 16);
    gpuVecGEMM<<<blockDims, threadDims>>>(a, W, b, z, N, M, H);
}


/*************************************************************************************/
/********************************** Sigmoid Kernel ***********************************/
/*************************************************************************************/

__global__
void gpuSigmoid(double* A, int M, int N) {
    const int row = threadIdx.x + blockDim.x * blockIdx.x;
    const int col = threadIdx.y + blockDim.y * blockIdx.y;
    if (row >= M || col >= N) return;

    A[col * M + row] = 1.0 / (1.0 + exp(-A[col * M + row]));
}

// assumes mat is column major storage and of dimensions M x N,
// performs in-place sigmoid function
int mySigmoid(double* mat, int M, int N) {
    // launch kernel
    dim3 threadDims(32, 24);
    dim3 blockDims((M + 31) / 32, (N + 23) / 24);
    gpuSigmoid<<<blockDims, threadDims>>>(mat, M, N);

    return 0;
}


/*************************************************************************************/
/********************************** Softmax Kernel ***********************************/
/*************************************************************************************/

__global__
void gpuExp(double* mat, int N, int C) {
    const int id = threadIdx.x + blockDim.x * blockIdx.x;
    const int c = threadIdx.y + blockDim.y * blockIdx.y;
    if (id >= N || c >= C) return;

    mat[c * N + id] = exp(mat[c * N + id]);
}

__global__
void gpuRowSums(const double* mat, double* sums, int N, int C) {
    const int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= N) return;

    sums[id] = 0.0;
    for (int i = 0; i < C; ++i) {
        sums[id] += mat[i * N + id];
    }
}

__global__
void gpuRowDivide(double* mat, const double* sums, int N, int C) {
    const int id = threadIdx.x + blockDim.x * blockIdx.x;
    const int c = threadIdx.y + blockDim.y * blockIdx.y;
    if (id >= N || c >= C) return;

    mat[c * N + id] = mat[c * N + id] / sums[id];
}

// assumes column major storage mat with dimension N x C and the class
// dimension extends on the columns, performs in-place softmax on mat
int mySoftmax(double* mat, int N, int C) {
    dim3 threadDims(32, 24);
    dim3 blockDims( (N + 31) / 32, (C + 23) / 24);
    // compute elementwise exp
    gpuExp<<<blockDims, threadDims>>>(mat, N, C);

    // compute row sums
    double *sums;
    checkCudaErrors(cudaMalloc((void**) &sums, N * sizeof(double)) );
    int threadDim1D = 768;
    int blockDim1D = (N + 767) / 768;
    gpuRowSums<<<blockDim1D, threadDim1D>>>(mat, sums, N, C);

    // compute softmax
    gpuRowDivide<<<threadDims, blockDims>>>(mat, sums, N, C);

    checkCudaErrors(cudaFree(sums));

    return 0;
}


/*************************************************************************************/
/************************ Elementwise Multiplication Kernel **************************/
/*************************************************************************************/

__global__
void gpuElemMult(const double* A, const double* B, double* C, int M, int N) {
    const int row = threadIdx.x + blockDim.x * blockIdx.x;
    const int col = threadIdx.y + blockDim.y * blockIdx.y;
    if (row >= M || col >= N) return;

    C[col * M + row] = A[col * M + row] * B[col * M + row] * (1.0 - B[col * M + row]);
}

// perform the elementwise multiplication operation that happens in two layer
// NN when computing the derivative of entropy on z1
void elemMults(const double* A, const double* B, double*& C, int M, int N) {
    checkCudaErrors( cudaMalloc((void**) &C, M * N * sizeof(double)) );

    dim3 threadDims(32, 24);
    dim3 blockDims( (M + 31) / 32, (N + 23) / 24);
    // compute elementwise multiply
    gpuElemMult<<<blockDims, threadDims>>>(A, B, C, M, N);
}


/*************************************************************************************/
/******************************** Column Sum Kernel **********************************/
/*************************************************************************************/

__global__
void gpuColSums(const double* mat, double* sums, int M, int N) {
    const int col = threadIdx.y + blockIdx.y * blockDim.y;
    const int id = threadIdx.x;
    if (col >= N) return;

    const int nPartials = 256;
    __shared__ double reductions[256];

    double partialSum = 0.0;
    for (int i = 0; i * nPartials + id < M; ++i) {
        partialSum += mat[col * M + i * nPartials + id];
    }
    reductions[id] = partialSum;

    __syncthreads();

    for (int i = 256 / 2; i > 0; i >>= 1) {
        if (id < i) {
            reductions[id] += reductions[id + i];
        }

        __syncthreads();
    }

    if (id == 0) {
        sums[col] = reductions[0];
    }
}

void columnSums(const double* mat, double* sums, int M, int N) {
    dim3 threadDims(256, 1);
    dim3 blockDims(1, N);
    // compute column sums
    gpuColSums<<<blockDims, threadDims>>>(mat, sums, M, N);
}


/*************************************************************************************/
/********************************* Addition Kernel ***********************************/
/*************************************************************************************/

__global__
void gpuMatAdd(double* X, const double* Y, double a, double b, int M, int N) {
    const int row = threadIdx.x + blockDim.x * blockIdx.x;
    const int col = threadIdx.y + blockDim.y * blockIdx.y;
    if (row >= M || col >= N) return;

    X[col * M + row] = a * X[col * M + row] + b * Y[col * M + row];
}

// performs the operation X = aX + bY on MxN matrices
void matAdd(double* X, double* Y, double a, double b, int M, int N) {
    // launch kernel
    dim3 threadDims(32, 24);
    dim3 blockDims((M + 31) / 32, (N + 23) / 24);
    gpuMatAdd<<<blockDims, threadDims>>>(X, Y, a, b, M, N);
}


__global__
void gpuVecAdd(double* X, const double* Y, double a, double b, int M) {
    const int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= M) return;

    X[tid] = a * X[tid] + b * Y[tid];
}

// performs the operation X = aX + bY on dim M vectors
void vecAdd(double* X, double* Y, double a, double b, int M) {
    // launch kernel
    dim3 threadDims(768);
    dim3 blockDims((M + 767) / 768);
    gpuVecAdd<<<blockDims, threadDims>>>(X, Y, a, b, M);
}


/*************************************************************************************/
/********************************* Transpose Kernel **********************************/
/*************************************************************************************/

const int warp_size = 32;
const int num_warps = 8;

__global__
void fastTranspose(double* array_in, double* array_out, int n_rows, int n_cols) {
    const int warp_id  = threadIdx.y;
    const int lane     = threadIdx.x;

    __shared__ double block[warp_size][warp_size + 1];

    int bc = blockIdx.x;
    int br = blockIdx.y;

    // Load 32x32 block into shared memory

    int gr = br * warp_size + lane;
    if (gr < n_rows) {
        for (int i = 0; i < 4; ++i) {
            int gc = bc * warp_size + i * num_warps + warp_id;
            if (gc >= n_cols) break;
            block[lane][i * num_warps + warp_id] = array_in[gc * n_rows + gr];
        }
    }

    __syncthreads();

    gr = bc * warp_size + lane;

    if (gr >= n_cols) return;
    for(int i = 0; i < 4; ++i) {
        int gc = br * warp_size + i * num_warps + warp_id;
        if (gc >= n_rows) return;
        array_out[gr + gc * n_cols] = block[i * num_warps + warp_id][lane];
    }
}

// transpose MxN matrix A into AT, both column major storage
void transpose(double* A, double*& AT, int M, int N) {
    // launch kernel

    checkCudaErrors( cudaMalloc((void**) &AT, M * N * sizeof(double)) );

    dim3 threadDims(warp_size, num_warps);
    dim3 blockDims((N + warp_size - 1) / warp_size, (M + warp_size - 1) / warp_size);

    fastTranspose<<<blockDims, threadDims>>>(A, AT, M, N);
}


/*************************************************************************************/
/*********************************** Free Function ***********************************/
/*************************************************************************************/

void freePtrs(std::vector<double*> ptrs) {
    for (unsigned int i = 0; i < ptrs.size(); ++i) {
        checkCudaErrors( cudaFree(ptrs[i]) );
    }
}


/*************************************************************************************/
/********************************** Copy Functions ***********************************/
/*************************************************************************************/

void copyToDevice(const double* hostPtr, double* devicePtr, unsigned int size) {
    checkCudaErrors (cudaMemcpy(devicePtr, hostPtr, size, cudaMemcpyHostToDevice));
}

int copyToDevice(const DataPointers& hostData, DataPointers& deviceData) {
    unsigned int xSize = hostData.N * hostData.H0 * sizeof(double);
    unsigned int ySize = hostData.N * hostData.C * sizeof(double);

    checkCudaErrors( cudaMalloc((void**) &(deviceData.XT), xSize) );
    checkCudaErrors( cudaMalloc((void**) &(deviceData.yT), ySize) );
    checkCudaErrors (cudaMemcpy(deviceData.XT, hostData.XT, xSize, cudaMemcpyHostToDevice));

    copyToDevice(hostData.yT, deviceData.yT, ySize);

    return 0;
}

int copyToDevice(const NNPointers& hostNN, NNPointers& deviceNN) {
    deviceNN.W.resize(hostNN.num_layers);
    deviceNN.b.resize(hostNN.num_layers);

    for (int i = 0; i < hostNN.num_layers; ++i) {
        double* deviceW;
        double* deviceB;
        unsigned int wSize = hostNN.H[i] * hostNN.H[i+1] * sizeof(double);
        unsigned int bSize = hostNN.H[i+1] * sizeof(double);

        checkCudaErrors( cudaMalloc((void**) &deviceW, wSize) );
        checkCudaErrors( cudaMalloc((void**) &deviceB, bSize) );

        copyToDevice(hostNN.W[i], deviceW, wSize);
        copyToDevice(hostNN.b[i], deviceB, bSize);

        deviceNN.W[i] = deviceW;
        deviceNN.b[i] = deviceB;
    }

    return 0;
}

int copyToDevice(const GradPtr& hostGrad, GradPtr& deviceGrad) {
    assert(deviceGrad.num_layers == hostGrad.num_layers);
    assert(!hostGrad.isDevice);
    assert(deviceGrad.isDevice);
    for (int i = 0; i < hostGrad.num_layers; ++i) {
        unsigned int wSize = hostGrad.H[i] * hostGrad.H[i+1] * sizeof(double);
        unsigned int bSize = hostGrad.H[i+1] * sizeof(double);
        copyToDevice(hostGrad.dW[i], deviceGrad.dW[i], wSize);
        copyToDevice(hostGrad.db[i], deviceGrad.db[i], bSize);
    }
    return 0;
}

void copyToHost(const double* devicePtr, double* hostPtr, unsigned int size) {
    checkCudaErrors (cudaMemcpy(hostPtr, devicePtr, size, cudaMemcpyDeviceToHost));
}

int copyToHost(const DataPointers& devicePtrs, DataPointers& hostPtrs) {
    copyToHost(devicePtrs.XT, hostPtrs.XT, devicePtrs.N * devicePtrs.H0 * sizeof(double));
    copyToHost(devicePtrs.yT, hostPtrs.yT, devicePtrs.N * devicePtrs.C * sizeof(double));
    return 0;
}

int copyToHost(const NNPointers& deviceNN, NNPointers& hostNN) {
    hostNN.W.resize(deviceNN.num_layers);
    hostNN.b.resize(deviceNN.num_layers);

    for (int i = 0; i < deviceNN.num_layers; ++i) {
        unsigned int wSize = deviceNN.H[i] * deviceNN.H[i+1] * sizeof(double);
        unsigned int bSize = deviceNN.H[i+1] * sizeof(double);
        double* hostW = (double*) malloc(wSize);
        double* hostB = (double*) malloc(bSize);

        copyToHost(deviceNN.W[i], hostW, wSize);
        copyToHost(deviceNN.b[i], hostB, bSize);

        hostNN.W[i] = hostW;
        hostNN.b[i] = hostB;
    }

    return 0;
}

int copyToHost(const GradPtr& deviceGrad, GradPtr& hostGrad) {
    assert(deviceGrad.num_layers == hostGrad.num_layers);
    assert(!hostGrad.isDevice);
    assert(deviceGrad.isDevice);
    for (int i = 0; i < deviceGrad.num_layers; ++i) {
        unsigned int wSize = deviceGrad.H[i] * deviceGrad.H[i+1] * sizeof(double);
        unsigned int bSize = deviceGrad.H[i+1] * sizeof(double);
        copyToHost(deviceGrad.dW[i], hostGrad.dW[i], wSize);
        copyToHost(deviceGrad.db[i], hostGrad.db[i], bSize);
    }
    return 0;
}

int copyToHost(const CachePtr& deviceCache, CachePtr& hostCache) {
    assert(!hostCache.isDevice);
    assert(deviceCache.isDevice);
    copyToHost(deviceCache.X, hostCache.X, deviceCache.N * deviceCache.H[0] * sizeof(double));
    copyToHost(deviceCache.a1, hostCache.a1, deviceCache.N * deviceCache.H[1] * sizeof(double));
    copyToHost(deviceCache.yDiff, hostCache.yDiff, deviceCache.N * deviceCache.H[2] * sizeof(double));
    return 0;
}
