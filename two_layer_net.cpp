#include "two_layer_net.h"

#include <armadillo>
#include <vector>
#include "utils/common.h"
#include "gpu_func.h"
#include "mpi.h"
#include <string.h>
#include "iomanip"

#define MPI_SAFE_CALL( call ) do {                               \
    int err = call;                                              \
    if (err != MPI_SUCCESS) {                                    \
        fprintf(stderr, "MPI error %d in file '%s' at line %i",  \
               err, __FILE__, __LINE__);                         \
        exit(1);                                                 \
    } } while(0)

double norms (TwoLayerNet &nn) {
      double norm_sum = 0;

      for (int i = 0; i < nn.num_layers; ++i)  {
        norm_sum += arma::accu (arma::square (nn.W[i]));
      }

      return norm_sum;
}

void write_cpudata_tofile(TwoLayerNet &nn, int iter)
{
  std::stringstream s;
  s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
  nn.W[0].save(s.str(), arma::raw_ascii);
  std::stringstream t;
  t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
  nn.W[1].save(t.str(), arma::raw_ascii);
  std::stringstream u;
  u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
  nn.b[0].save(u.str(), arma::raw_ascii);
  std::stringstream v;
  v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
  nn.b[1].save(v.str(), arma::raw_ascii);  
}

void write_diff_gpu_cpu(TwoLayerNet &nn, int iter, std::ofstream& error_file)
{
  arma::mat A, B, C, D; 

  std::stringstream s;
  s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
  A.load(s.str(), arma::raw_ascii);
  double max_errW0 = arma::norm(nn.W[0]-A, "inf")/arma::norm(A, "inf");
  double L2_errW0  = arma::norm(nn.W[0]-A,2)/arma::norm(A,2);

  std::stringstream t;
  t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
  B.load(t.str(), arma::raw_ascii);
  double max_errW1 = arma::norm(nn.W[1]-B, "inf")/arma::norm(B, "inf");
  double L2_errW1  = arma::norm(nn.W[1]-B,2)/arma::norm(B,2);

  std::stringstream u;
  u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
  C.load(u.str(), arma::raw_ascii);
  double max_errb0 = arma::norm(nn.b[0]-C, "inf")/arma::norm(C, "inf");
  double L2_errb0  = arma::norm(nn.b[0]-C,2)/arma::norm(C,2);

  std::stringstream v;
  v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
  D.load(v.str(), arma::raw_ascii);
  double max_errb1 = arma::norm(nn.b[1]-D, "inf")/arma::norm(D, "inf");
  double L2_errb1  = arma::norm(nn.b[1]-D,2)/arma::norm(D,2);

  int ow = 15;
  if( iter == 0 ) {
    error_file << std::left<< std::setw(ow) << "Iteration" << std::left<< std::setw(ow) << "Max Err W0" << std::left << std::setw(ow) << "Max Err W1" 
    << std::left<< std::setw(ow) << "Max Err b0" << std::left<< std::setw(ow) << "Max Err b1" << std::left << std::setw(ow) << "L2 Err W0" << std::left 
    << std::setw(ow) << "L2 Err W1" << std::left<< std::setw(ow) << "L2 Err b0" << std::left<< std::setw(ow) << "L2 Err b1" << "\n";
  }
  error_file << std::left << std::setw(ow) << iter << std::left << std::setw(ow) << max_errW0 << std::left << std::setw(ow) << max_errW1 << 
  std::left << std::setw(ow) << max_errb0 << std::left << std::setw(ow) << max_errb1 << std::left<< std::setw(ow) << L2_errW0 << std::left << 
  std::setw(ow) << L2_errW1 << std::left << std::setw(ow) << L2_errb0 << std::left<< std::setw(ow) << L2_errb1 << "\n";
  
} 


void feedforward (TwoLayerNet &nn, const arma::mat& X, struct cache& cache)
{
  cache.z.resize(2);
  cache.a.resize(2);

  // std::cout << W[0].n_rows << "\n";
  assert (X.n_cols == nn.W[0].n_cols);
  cache.X = X;
  int N = X.n_rows;

  arma::mat z1 = X * nn.W[0].t() + arma::repmat(nn.b[0], N, 1);
  cache.z[0] = z1;

  // std::cout << "Computing a1 " << "\n";
  arma::mat a1;
  sigmoid (z1, a1);
  cache.a[0] = a1;

  // std::cout << "Computing z2 " << "\n";
  assert (a1.n_cols == nn.W[1].n_cols);
  arma::mat z2 = a1 * nn.W[1].t() + arma::repmat(nn.b[1], N, 1);
  cache.z[1] = z2;

  // std::cout << "Computing a2 " << "\n";
  arma::mat a2;
  softmax (z2, a2);
  cache.a[1] = cache.yc = a2;
}

/*
 * Computes the gradients of the cost w.r.t each param.
 * MUST be called after feedforward since it uses the bpcache.
 * @params y : N x C one-hot row vectors
 * @params bpcache : Output of feedforward.
 * @params bpgrads: Returns the gradients for each param
 */
void backprop (TwoLayerNet &nn, const arma::mat& y, double reg, const struct cache& bpcache, struct grads& bpgrads)
{
  bpgrads.dW.resize(2);
  bpgrads.db.resize(2);
  int N = y.n_rows;

  // std::cout << "backprop " << bpcache.yc << "\n";
  arma::mat diff = (1.0 / N) * (bpcache.yc - y);
  bpgrads.dW[1] = diff.t() * bpcache.a[0] + reg * nn.W[1];
  bpgrads.db[1] = arma::sum (diff, 0);
  arma::mat da1 = diff * nn.W[1];

  arma::mat dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);

  bpgrads.dW[0] = dz1.t() * bpcache.X + reg * nn.W[0];
  bpgrads.db[0] = arma::sum(dz1, 0);
}

/*
 * Computes the Cross-Entropy loss function for the neural network.
 */
double loss (TwoLayerNet &nn, const arma::mat& yc, const arma::mat& y, double reg)
{
  int N = yc.n_rows;
  double ce_sum = -arma::accu (arma::log (yc.elem (arma::find (y == 1))));

  double data_loss = ce_sum / N;
  double reg_loss = 0.5 * reg * norms(nn);
  double loss = data_loss + reg_loss;
  // std::cout << "Loss: " << loss << "\n";
  return loss;
}

/*
 * Returns a vector of labels for each row vector in the input
 */
void predict (TwoLayerNet &nn, const arma::mat& X, arma::mat& label)
{
  struct cache fcache;
  feedforward (nn, X, fcache);
  label.set_size (X.n_rows);

  for (int i = 0; i < X.n_rows; ++i) {
    arma::uword row, col;
    fcache.yc.row(i).max (row, col);
    label(i) = col;
  }
}

/* 
 * Computes the numerical gradient
 */
void numgrad (TwoLayerNet &nn, const arma::mat& X, const arma::mat& y, double reg, struct grads& numgrads)
{
  double h = 0.00001;
  struct cache numcache;
  numgrads.dW.resize(nn.num_layers);
  numgrads.db.resize(nn.num_layers);

  for (int i = 0; i < nn.num_layers; ++i) {
    numgrads.dW[i].resize (nn.W[i].n_rows, nn.W[i].n_cols);
    for (int j = 0; j < nn.W[i].n_rows; ++j) {
      for (int k = 0; k < nn.W[i].n_cols; ++k) {
        double oldval = nn.W[i](j,k);
        nn.W[i](j, k) = oldval + h;
        feedforward (nn, X, numcache);
        double fxph = loss (nn, numcache.yc, y, reg);
        nn.W[i](j, k) = oldval - h;
        feedforward (nn, X, numcache);
        double fxnh = loss (nn, numcache.yc, y, reg);
        numgrads.dW[i](j, k) = (fxph - fxnh) / (2*h);
        nn.W[i](j, k) = oldval;
      }
    }
  }

   for (int i = 0; i < nn.num_layers; ++i) {
    numgrads.db[i].resize (nn.b[i].n_rows, nn.b[i].n_cols);
    for (int j = 0; j < nn.b[i].size(); ++j) {
      double oldval = nn.b[i](j);
      nn.b[i](j) = oldval + h;
      feedforward (nn, X, numcache);
      double fxph = loss (nn, numcache.yc, y, reg);
      nn.b[i](j) = oldval - h;
      feedforward (nn, X, numcache);
      double fxnh = loss (nn, numcache.yc, y, reg);
      numgrads.db[i](j) = (fxph - fxnh) / (2*h);
      nn.b[i](j) = oldval;
    }
  }
}

/*
 * Train the neural network &nn
 */
void train (TwoLayerNet &nn, const arma::mat& X, const arma::mat& y, double learning_rate, double reg, 
    const int epochs, const int batch_size, bool grad_check, int print_every, int debug)
{
  int N = X.n_rows;
  int iter = 0;
  int print_flag = 0;

  for (int epoch = 0 ; epoch < epochs; ++epoch) {
    int num_batches = (int) ceil ( N / (float) batch_size);    

    for (int batch = 0; batch < num_batches; ++batch) {
      int last_row = std::min ((batch + 1)*batch_size-1, N-1);
      arma::mat X_batch = X.rows (batch * batch_size, last_row);
      arma::mat y_batch = y.rows (batch * batch_size, last_row);

      struct cache bpcache;
      feedforward (nn, X_batch, bpcache);
      
      struct grads bpgrads;
      backprop (nn, y_batch, reg, bpcache, bpgrads);

      if (print_every > 0 && iter % print_every == 0) {
       if (grad_check) {
          struct grads numgrads;
          numgrad (nn, X_batch, y_batch, reg, numgrads);
          assert (gradcheck (numgrads, bpgrads));
        }
        std::cout << "Loss at iteration " << iter << " of epoch " << epoch << "/" << epochs << " = " << loss (nn, bpcache.yc, y_batch, reg) << "\n";
      }

      // Gradient descent step
      for (int i = 0; i < nn.W.size(); ++i) {
        nn.W[i] -= learning_rate * bpgrads.dW[i];
      }

      for (int i = 0; i < nn.b.size(); ++i) {
        nn.b[i] -= learning_rate * bpgrads.db[i];
      }

      /* Debug routine runs only when debug flag is set. If print_every is zero, it saves
         for the first batch of each epoch to avoid saving too many large files.
         Note that for the first time, you have to run debug and serial modes together.
         This will run the following function and write out files to CPUmats folder.
         In the later runs (with same parameters), you can use just the debug flag to 
         output diff b/w CPU and GPU without running CPU version */
      if(print_every <= 0)
        print_flag = batch == 0;
      else
        print_flag = iter % print_every == 0;

      if(debug && print_flag)
        write_cpudata_tofile(nn, iter);

      iter++;
    }
  }    
}


/***************************************************************************************/
/******************************** Parallel Implementation ******************************/
/***************************************************************************************/

void myFeedforward(const NNPointers& nn, double* XT, CachePtr& cache) {
    std::vector<double*> tempPtrs;

    cache.X = XT;
    double* X;
    transpose(XT, X, cache.H[0], cache.N);
    tempPtrs.push_back(X);

    double* a1;
    computeZ(X, nn.W[0], nn.b[0], a1, cache.N, cache.H[0], cache.H[1]);
    mySigmoid(a1, cache.N, cache.H[1]);
    cache.a1 = a1;

    double* yc;
    computeZ(a1, nn.W[1], nn.b[1], yc, cache.N, cache.H[1], cache.H[2]);
    mySoftmax(yc, cache.N, cache.H[2]);
    cache.yDiff = yc;

    freePtrs(tempPtrs);
}

void myBackprop(const NNPointers& nn, double* yT, double reg, const CachePtr& cache, GradPtr& deviceGrad, int totalSize, int num_procs) {

    std::vector<double*> tempPtrs;

    // compute yDiff, an NxC matrix
    double* y;
    transpose(yT, y, cache.H[2], cache.N);
    matAdd(cache.yDiff, y, 1.0 / totalSize, -1.0 / totalSize, cache.N, cache.H[2]);
    tempPtrs.push_back(y);

    // compute dW2 = a1^T * yDiff + reg * W2
    double* a1T;
    transpose(cache.a1, a1T, cache.N, cache.H[1]);
    myAllocGEMM(a1T, cache.yDiff, nn.W[1], deviceGrad.dW[1], 1.0, reg / num_procs, cache.H[1], cache.H[2], cache.N, false);
    tempPtrs.push_back(a1T);

    // compute db2 = colSum (yDiff)
    columnSums(cache.yDiff, deviceGrad.db[1], cache.N, cache.H[2]);

    // compute gradients for W1 and b1
    // first, compute dCE/da1 = yDiff * W2T
    double* da1;
    double* W1T;
    transpose(nn.W[1], W1T, cache.H[1], cache.H[2]);
    myAllocGEMM(cache.yDiff, W1T, cache.a1, da1, 1.0, 0.0, cache.N, cache.H[1], cache.H[2], true);
    tempPtrs.push_back(W1T);
    tempPtrs.push_back(da1);

    // then compute dCE/dz1 = da1 o a1 o (1 - a1)
    double* dz1;
    elemMults(da1, cache.a1, dz1, cache.N, cache.H[1]);
    tempPtrs.push_back(dz1);

    // lastly, dW1 = X^T * dz1 + reg * W1
    myAllocGEMM(cache.X, dz1, nn.W[0], deviceGrad.dW[0], 1.0, reg / num_procs, cache.H[0], cache.H[1], cache.N, false);

    // db1 = colSums (dz1)
    columnSums(dz1, deviceGrad.db[0], cache.N, cache.H[1]);

    freePtrs(tempPtrs);
}

void divideBatch(int* offsets, int* sizes, int totalSize, int num_procs, int batch, int batch_size) {
    double rankSize = totalSize / num_procs;
    int offset = batch * batch_size;
    int cumulSize = 0;
    for (int i = 0; i < num_procs; ++i) {
        offsets[i] = offset;
        sizes[i] = (int) (rankSize * (i+1)) - cumulSize;
        offset += sizes[i];
        cumulSize += sizes[i];
    }
    assert(cumulSize == totalSize);
}

/*
 * Train the neural network &nn of rank 0 in parallel. Your MPI implementation 
 * should mainly be in this function.
 */
void parallel_train (TwoLayerNet &nn, const arma::mat& X, const arma::mat& y, double learning_rate, double reg, 
    const int epochs, const int batch_size, bool grad_check, int print_every, int debug)
{
  int rank, num_procs;
  MPI_SAFE_CALL (MPI_Comm_size (MPI_COMM_WORLD, &num_procs));
  MPI_SAFE_CALL (MPI_Comm_rank (MPI_COMM_WORLD, &rank));

  int N = (rank == 0)?X.n_rows:0;
  MPI_SAFE_CALL (MPI_Bcast (&N, 1, MPI_INT, 0, MPI_COMM_WORLD));

  std::ofstream error_file;
  error_file.open("Outputs/CpuGpuDiff.txt");
  int print_flag = 0;

  /* HINT: You can obtain a raw pointer to the memory used by Armadillo Matrices
     for storing elements in a column major way. Or you can allocate your own array
     memory space and store the elements in a row major way. Remember to update the
     Armadillo matrices in TwoLayerNet &nn of rank 0 before returning from the function. */
  DataPointers hostData(N, nn.H[0], nn.H[nn.num_layers], false);
  arma::mat XT = X.t();
  arma::mat yT = y.t();
  if (rank == 0) {
    memcpy(hostData.XT, XT.memptr(), N * nn.H[0] * sizeof(double));
    memcpy(hostData.yT, yT.memptr(), N * nn.H[2] * sizeof(double));
  }
  MPI_SAFE_CALL (MPI_Bcast (hostData.XT, N * nn.H[0], MPI_DOUBLE, 0, MPI_COMM_WORLD));
  MPI_SAFE_CALL (MPI_Bcast (hostData.yT, N * nn.H[nn.num_layers], MPI_DOUBLE, 0, MPI_COMM_WORLD));
  // if (rank == 0) std::cout << "Bcast XT and yT" << std::endl;

  NNPointers hostNN(nn.H, false);
  for (int i = 0; i < nn.num_layers; ++i) {
    arma::mat WT = nn.W[i].t();
    if (rank == 0) {
      memcpy(hostNN.W[i], WT.memptr(), nn.H[i] * nn.H[i+1] * sizeof(double));
      memcpy(hostNN.b[i], nn.b[i].memptr(), nn.H[i+1] * sizeof(double));
    }
    MPI_SAFE_CALL (MPI_Bcast (hostNN.W[i], nn.H[i] * nn.H[i+1], MPI_DOUBLE, 0, MPI_COMM_WORLD));
    MPI_SAFE_CALL (MPI_Bcast (hostNN.b[i], nn.H[i+1], MPI_DOUBLE, 0, MPI_COMM_WORLD));
  }
  // if (rank == 0) std::cout << "Bcast W and b" << std::endl;

  DataPointers deviceData(N, nn.H[0], nn.H[nn.num_layers], true);
  NNPointers deviceNN(nn.H, true);
  int success = copyToDevice(hostData, deviceData) + copyToDevice(hostNN, deviceNN);
  if (success != 0) {
    std::cout << "copyToDevice failed" << std::endl;
  }

  /* iter is a variable used to manage debugging. It increments in the inner loop
     and therefore goes from 0 to epochs*num_batches */
  int iter = 0;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    int num_batches = (N + batch_size - 1)/batch_size;
    for (int batch = 0; batch < num_batches; ++batch) {
      /* TODO:
       * Possible Implementation:
       * 1. subdivide input batch of images and `MPI_scatter()' to each MPI node
       * 2. compute each sub-batch of images' contribution to network coefficient updates
       * 3. reduce the coefficient updates and broadcast to all nodes with `MPI_Allreduce()'
       * 4. update local network coefficient at each node
       */

      int offsets[num_procs];
      int sizes[num_procs];

      int last_row = std::min ((batch + 1)*batch_size-1, N-1);
      int totalSize = last_row - batch * batch_size + 1;

      if (rank == 0) {
        divideBatch(&offsets[0], &sizes[0], totalSize, num_procs, batch, batch_size);
      }

      int size;
      int offset;

      MPI_Scatter(&offsets[0], 1, MPI_INT, &offset, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Scatter(&sizes[0], 1, MPI_INT, &size, 1, MPI_INT, 0, MPI_COMM_WORLD);

      CachePtr deviceCache(nn.H, size, true);
      myFeedforward (deviceNN, deviceData.XT + nn.H[0] * offset, deviceCache);

      GradPtr deviceGrad(nn.H, true);
      myBackprop (deviceNN, deviceData.yT + nn.H[2] * offset, reg, deviceCache, deviceGrad, totalSize, num_procs);

      GradPtr hostGrad(nn.H, false);
      copyToHost(deviceGrad, hostGrad);

      GradPtr globalHostGrad(nn.H, false);
      // MPI_Allreduce on hostGrad
      for (int i = 0; i < nn.num_layers; ++i) {
        MPI_SAFE_CALL ( MPI_Allreduce(hostGrad.dW[i], globalHostGrad.dW[i], nn.H[i] * nn.H[i+1], MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) );
        MPI_SAFE_CALL ( MPI_Allreduce(hostGrad.db[i], globalHostGrad.db[i], nn.H[i+1], MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) );
      }

      copyToDevice(globalHostGrad, deviceGrad);

      for (int i = 0; i < deviceNN.num_layers; ++i) {
          matAdd(deviceNN.W[i], deviceGrad.dW[i], 1.0, -learning_rate, deviceNN.H[i], deviceNN.H[i+1]);
          vecAdd(deviceNN.b[i], deviceGrad.db[i], 1.0, -learning_rate, deviceNN.H[i+1]);
      }

      // END TODO

      iter++;
    }
  }
  copyToHost(deviceNN, hostNN);
  for (int i = 0; i < nn.num_layers; ++i) {
    arma::mat WT(hostNN.W[i], hostNN.H[i], hostNN.H[i+1], true);
    arma::mat W = WT.t();
    nn.W[i] = W;
    arma::mat bMat(hostNN.b[i], 1, hostNN.H[i+1], true);
    arma::rowvec b(bMat);
    nn.b[i] = b;
  }

  error_file.close();
}