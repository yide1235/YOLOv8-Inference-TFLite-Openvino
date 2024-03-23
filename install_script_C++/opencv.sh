#!/bin/bash
# Install minimal prerequisites (Ubuntu 18.04 as reference)
sudo apt update && sudo apt install -y cmake g++ wget unzip
# Download and unpack sources
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
unzip opencv.zip
# Create build directory
mkdir -p build && cd build
# Configure
cmake -DCMAKE_INSTALL_PREFIX=/usr ../opencv-4.x
# Build
cmake --build . -j$(nproc)
# Install
sudo make install
