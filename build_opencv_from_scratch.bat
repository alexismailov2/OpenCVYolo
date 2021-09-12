git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 4.5.2
cd ..

git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout 4.5.2
cd ..

cd opencv
mkdir build
cd build

cmake.exe -D CMAKE_BUILD_TYPE=RELEASE ^
 -D CMAKE_INSTALL_PREFIX=./opencv_4_5_2_cuda_release ^
 -D INSTALL_C_EXAMPLES=OFF ^
 -D WITH_TBB=ON ^
 -D WITH_CUDA=ON ^
 -D BUILD_opencv_cudacodec=OFF ^
 -D ENABLE_FAST_MATH=1 ^
 -D CUDA_FAST_MATH=1 ^
 -D WITH_CUBLAS=1 ^
 -D WITH_CUDNN=ON ^
 -D OPENCV_DNN_CUDA=ON ^
 -D CUDA_ARCH_BIN=6.1 ^
 -D WITH_V4L=ON ^
 -D WITH_QT=OFF ^
 -D WITH_OPENGL=ON ^
 -D WITH_GSTREAMER=ON ^
 -D OPENCV_GENERATE_PKGCONFIG=ON ^
 -D OPENCV_PC_FILE_NAME=opencv.pc ^
 -D OPENCV_ENABLE_NONFREE=ON ^
 -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules ^
 -D BUILD_EXAMPLES=OFF ^
 -D BUILD_PERF_TESTS=OFF ^
 -D WITH_PYTHON=OFF ^
 -D WITH_JAVA=OFF ^
 -D BUILD_TESTS=OFF ..
cmake.exe --build build --target all