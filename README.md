# OpenCVYolo
This is the simple demo of usage Yolo with OpenCV.

#Dependencies
- OpenCV 4.2.0 and higher preferable for CUDA backend usage.
- gcc/clang with c++17 support and higher.
- download models and data from https://mega.nz/folder/a24hzQZK#KxVDB19Pf2d-mS6GMidbAg

#Tested
- Desktop Ubuntu 18.04
- Nvidia jetson nano
- MacOS

#Build
Install opencv(for jetson nano there is a package which was prebuilt with CUDA support in the DNN module).
Just run ./build.sh

#Run
Run ./run.sh

Or you can use any your custom video/camera/video stream:
./build/opencv_yolo <path to your video/camera device/videostream>

