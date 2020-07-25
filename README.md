# stop-covoid19
This is the simple demo of the people detection, and visualization danger distances between them.

#Dependencies
- OpenCV 4.2.0 and higher preferable for CUDA backend usage.
- gcc/clang with c++14 support and higher.
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
./build/yolov3_opencv <path to your video/camera device/videostream>

