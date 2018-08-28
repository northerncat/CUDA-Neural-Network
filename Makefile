CC=mpic++
CUD=nvcc
CFLAGS=-O2 -std=c++11 -I./lib/usr/include
LFLAGS= -O2 -larmadillo -lcublas -lcudart -L=./lib/usr/lib64 -Wl,-rpath=./lib/usr/lib64
CUDFLAGS=-O2 -c -arch=sm_20 -Xcompiler -Wall,-Winline,-Wextra,-Wno-strict-aliasing

main: main.o two_layer_net.o mnist.o common.o gpu_func.o tests.o
	$(CC) $(LFLAGS) main.o two_layer_net.o mnist.o common.o gpu_func.o tests.o -o main

main.o: main.cpp
	$(CC) $(CFLAGS) -c main.cpp

two_layer_net.o: two_layer_net.cpp utils/test_utils.h two_layer_net.h
	$(CC) $(CFLAGS) -c two_layer_net.cpp

mnist.o: utils/mnist.cpp
	$(CC) $(CFLAGS) -c utils/mnist.cpp

common.o: utils/common.cpp
	$(CC) $(CFLAGS) -c utils/common.cpp

tests.o: utils/tests.cpp utils/tests.h
	$(CC) $(CFLAGS) -c utils/tests.cpp

gpu_func.o: gpu_func.cu
	$(CUD) $(CUDFLAGS) -c gpu_func.cu

clean:
	rm -rf *.o main

