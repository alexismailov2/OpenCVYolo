#include <opencv_yolo/YOLOv3.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <vector>

#ifdef _MSC_VER
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#include "../src/TimeMeasuring.hpp"

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
   if (argc != 6)
   {
      std::cout << "Should be provided video file as input!" << std::endl;
      return 0;
   }

   auto yolov3 = YOLOv3{argv[2],
                        argv[3],
                        argv[4],
                        cv::Size{atoi(argv[5]), atoi(argv[5])}, 0.3f, 0.3f};

   static const std::string kWinName = "OpenCV YOLOv3 Demo";
   cv::namedWindow(kWinName, cv::WINDOW_NORMAL);

   if (fs::is_directory(argv[1]))
   {
      cv::VideoWriter video("outcpp.avi",
                            cv::VideoWriter::fourcc('M','J','P','G'),
                            10, cv::Size(atoi(argv[5]), atoi(argv[5])));

      for (auto file : fs::directory_iterator(argv[1]))
      {
         auto filePath = file.path().string();
         cv::Mat frame = cv::imread(filePath);
         if (frame.empty() || (cv::waitKey(1) == 27))
         {
            break;
         }
         draw(frame, yolov3.performPrediction(frame), cv::Scalar{0x00, 0xFF, 0x00}, 2);
         imshow(kWinName, frame);
         cv::resize(frame, frame, cv::Size(atoi(argv[5]), atoi(argv[5])));
         video.write(frame);
         cv::waitKey(100);
      }
      video.release();
      cv::destroyAllWindows();
      return 0;
   }

   cv::VideoCapture cap;
   cap.open(argv[1]);

   cv::VideoWriter video("outcpp.avi",
                         cv::VideoWriter::fourcc('M','J','P','G'),
                         10,
                         cv::Size(static_cast<int32_t>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
                                  static_cast<int32_t>(cap.get(cv::CAP_PROP_FRAME_HEIGHT))));

   cv::Mat frame;
   while (cv::waitKey(1) < 0)
   {
      TAKEN_TIME();
      cap >> frame;
      if (frame.empty() || (cv::waitKey(1) == 27))
      {
         break;
      }
      draw(frame, yolov3.performPrediction(frame), cv::Scalar{0x00, 0xFF, 0x00}, 2);
      video.write(frame);
      imshow(kWinName, frame);
   }

   cv::destroyAllWindows();
   return 0;
}
