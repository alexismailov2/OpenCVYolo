cmake_minimum_required(VERSION 3.14)
project(opencv_yolo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_VERBOSE_MAKEFILE ON)

if("${CUSTOM_OPENCV_BUILD_PATH}" STREQUAL "")
   find_package(OpenCV REQUIRED)
else()
   find_package(OpenCV REQUIRED PATHS ${CUSTOM_OPENCV_BUILD_PATH} NO_DEFAULT_PATH)
endif()

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME}
   YOLOv3.hpp
   YOLOv3.cpp
   TimeMeasuring.hpp
   yolov3-opencv.cpp)

message(STATUS OpenCV_LIBS=${OpenCV_LIBS})

target_link_libraries(${PROJECT_NAME}
   ${OpenCV_LIBS}
   stdc++fs)