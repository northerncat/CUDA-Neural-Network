# Neural Networks on CUDA
This project parallelizes the training phase of a three-layer neural network through CUDA. Besides implementing most of the algebraic operations in CUDA, two types of optimization is explored in this project: accelerated matrix operation with GPU and parallel training through the Message Passing Interface (MPI).

### GPU Accelerated Matrix Operation
Variations of neural networks rely on similar linear algebra operations known as the Basic Linear Algebra Subprograms (BLAS). The General Matrix Multiplication (GEMM) operation is one of them, being [applicable to fully connected layers, convolutional layers and many others alike](https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/). As a result, in this project, I optimized the performance of the GEMM operation through the use of blocking and shared memory on the GPU. With the aid of the NVidia Visual Profiler (NVVP), I was able to identify bottlenecks in the GEMM optimization and improve the performance of the kernel iteratively, until my implementation was close to the performance of the [cuBLAS kernel released by NVidia](https://developer.nvidia.com/cublas).

### Parallel Training with MPI
The [Message Passing Interface (MPI)](http://mpitutorial.com/tutorials/mpi-introduction/) is a high performance computing standard that allows multiple computers and/or GPUs to run a parallel program via exchanging messages. This is particularly beneficial for training a neural network, since the amount of data required to train a sophisticated neural network is large and even batch training on a single device can be resource consuming. In this project, I distributed batches of the input dataset to multiple MPI nodes, or GPUs on the same machine, to perform parallel training, while maintaing communication of neural network parameters among nodes. This would introduce communication overhead, but the performance gain through parallelization was still observable.

### Implementation and Results
See the writeup reports ([part 1](prelim_report.pdf) and [part 2](final_report.pdf)) I had for a detailed discussion of the implementation and results! To highligh a few key achievements:

- Achieved comparable performance on the GEMM operation kernel with the cuBLAS kernel from NVidia
- Profiled the performance of the entire neural network with the NVidia Visual Profiler and iteratively optimized expensive kernels
- Applied MPI to batch the dataset and enable training in parallel on multiple GPUs