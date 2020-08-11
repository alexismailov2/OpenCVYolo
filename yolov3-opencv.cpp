#include "YOLOv3.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/dnn/layer.details.hpp>

#include <iostream>
#include <vector>
#include <experimental/filesystem>

#include "TimeMeasuring.hpp"

#define FULL_YOLOV3 1
#define FULL_YOLOV4 0 // unfourtunately does not work because OpenCV does not support mish activation

namespace {
void draw(cv::Mat& frame,
          YOLOv3::Item::List const& list,
          cv::Scalar color,
          int32_t thickness)
{
   TAKEN_TIME();
   for (int i = 0; i < list.size(); ++i)
   {
       cv::rectangle(frame, list[i].boundingBox, color, thickness);
   }
}
} /// end namespace anonymous

auto main(int argc, char** argv) -> int32_t
{
   if (argc != 2)
   {
      std::cout << "Should be provided video file as input!" << std::endl;
      return 0;
   }

   static const std::string kWinName = "Stop COVOID19! Make this world heatlhy!!!";
   cv::namedWindow(kWinName, cv::WINDOW_NORMAL);

   cv::VideoCapture cap;
   cap.open(argv[1]);

#if FULL_YOLOV3
//   auto yolov3 = YOLOv3{"./models/standard/yolov3.cfg",
//                        "./models/standard/yolov3.weights",
//                        "./models/standard/coco.names",
//                        cv::Size{608, 608}, 0.3f, 0.3f};

   // Workaround for tensorflow
   CV_DNN_REGISTER_LAYER_CLASS(LeakyRelu, cv::dnn::dnn4_v20191202::ReLULayer)
//   auto yolov3 = YOLOv3{"./models/kangaroo_20.pb",
//                        "./models/kangaroo_20.names",
//                        {{38,134, 47,286, 69,51}, {78,322, 84,169, 118,352}, {126,105, 140,215, 186,336}},
//                        cv::Size{1024, 1024}, 0.3f, 0.3f};

   auto yolov3 = YOLOv3{"./models/voc.pb",
                        "./models/voc.names",
                        {{24,34, 46,84, 68,185}, {116,286, 122,97, 171,180}, {214,327, 326,193, 359,359}},
                        cv::Size{480, 480}, 0.3f, 0.3f};


#elif FULL_YOLOV4
   auto yolov3 = YOLOv3{"./models/yolov4/yolov4.cfg",
                        "./models/yolov4/yolov4.weights",
                        "./models/yolov4/coco.names",
                        cv::Size{608, 608}, 0.3f, 0.3f};
#else
   auto yolov3 = YOLOv3{"./models/wilderperson/yolov3-tiny.cfg",
                        "./models/wilderperson/yolov3-tiny_14000.weights",
                        "./models/wilderperson/_.names",
                        cv::Size{416, 416}, 0.3f, 0.3f};
#endif

//   for (auto file : std::experimental::filesystem::directory_iterator("/home/oleksandr_ismailov/WORK/ADAS/OpenCVUNet/train/data2/imgs"))
//   {
//      auto filePath = file.path().string();
//      cv::Mat frame = cv::imread(filePath);
//      draw(frame, yolov3.performPrediction(frame), cv::Scalar{0x00, 0xFF, 0x00}, 2);
//      imshow(kWinName, frame);
//      cv::waitKey(1000);
//   }
//   return 0;
   cv::Mat frame;
   while (cv::waitKey(1) < 0)
   {
      TAKEN_TIME();
      cap >> frame;
      if (frame.empty())
      {
         cv::waitKey();
         break;
      }
      draw(frame, yolov3.performPrediction(frame), cv::Scalar{0x00, 0xFF, 0x00}, 2);
      imshow(kWinName, frame);
   }
   return 0;
}
