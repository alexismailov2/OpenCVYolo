#include <opencv_yolo/YOLOv3.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <vector>

#include "../../TimeMeasuring.hpp"

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

   auto yolov3 = YOLOv3{"./models/standard/yolov3.cfg",
                        "./models/standard/yolov3.weights",
                        "./models/standard/coco.names",
                        cv::Size{608, 608}, 0.3f, 0.3f};

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
