#!/bin/bash

# build pybind11
mkdir -p extern/pybind11/build
cd extern/pybind11/build
mkdir -p ~/.include
export pybind11_DIR=/mnt/irisgpfs/users/qlao/miniconda/envs/tf-gpu/lib/python3.9/site-packages/pybind11/share/cmake/pybind11
cmake .. -DCMAKE_INSTALL_PREFIX=~/.include
make install

# build solution
cd ../../..
mkdir -p build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=~/.include
make
cd ..
