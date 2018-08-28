#!/bin/bash

TRAIN_IMG_FILE_NAME="train-images-idx3-ubyte.gz"
TRAIN_LABELS_FILE_NAME="train-labels-idx1-ubyte.gz"

TEST_IMG_FILE_NAME="t10k-images-idx3-ubyte.gz"

rm -rf data/
mkdir data
cd data

echo "Downloading training images..."
wget -O "$TRAIN_IMG_FILE_NAME" "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
gunzip "$TRAIN_IMG_FILE_NAME"

echo "Downloading training labels..."
wget -O "$TRAIN_LABELS_FILE_NAME" "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
gunzip "$TRAIN_LABELS_FILE_NAME"

echo "Downloading test images..."
wget -O "$TEST_IMG_FILE_NAME" "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
gunzip "$TEST_IMG_FILE_NAME"

rm *.gz
cd ..

echo "Downloading Armadillo Library..."
wget http://sourceforge.net/projects/arma/files/armadillo-6.700.6.tar.gz
gunzip armadillo-6.700.6.tar.gz
tar -xf armadillo-6.700.6.tar
cd armadillo-6.700.6
./configure
make
make install DESTDIR=../lib/
cd ..
rm -rf armadillo-6.700.6*