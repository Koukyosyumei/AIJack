#!/bin/bash

# clean up
rm -rf build/*

# build
./script/build.sh

# test for c++
cd build
ctest -V
cd ..
