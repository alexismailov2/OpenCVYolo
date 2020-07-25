#include "YOLOv3.hpp"

#include "TimeMeasuring.hpp"

#include <opencv2/imgproc.hpp>

#include <iostream>
#include <vector>
#include <fstream>

namespace {

auto readClasses(std::string const& filename) -> std::vector<std::string>
{
   std::vector<std::string> classes;
   std::string className;
   auto fileWithClasses{std::ifstream(filename)};
   while (std::getline(fileWithClasses, className))
   {
      if (!className.empty())
      {
         classes.push_back(className);
      }
   }
   return classes;
}

} /// end namespace anonymous

YOLOv3::YOLOv3(std::string const& modelFile,
               std::string const& weightsFile,
               std::string const& classesFile,
               cv::Size inputSize,
               float confThreshold,
               float nmsThreshold)
   : _classes{readClasses(classesFile)}
   , _inputSize{inputSize}
   , _confThreshold{confThreshold}
   , _nmsThreshold{nmsThreshold}
   , _net{cv::dnn::readNetFromDarknet(modelFile, weightsFile)}
{
   std::cout << "Loaded config file: " << modelFile << std::endl;
   _net.setPreferableBackend(::cv::dnn::DNN_BACKEND_CUDA);
   _net.setPreferableTarget(::cv::dnn::DNN_TARGET_CUDA);
//   _net.setPreferableBackend(::cv::dnn::DNN_BACKEND_DEFAULT);
//   _net.setPreferableTarget(::cv::dnn::DNN_TARGET_CPU);
}

auto YOLOv3::performPrediction(cv::Mat const &frame,
                               std::function<bool(std::string const&)>&& filter,
                               bool isNeededToBeSwappedRAndB) -> Item::List
{
    TAKEN_TIME();
   _net.setInput(::cv::dnn::blobFromImage(frame, 1.0f / 255.0f, _inputSize, {}, isNeededToBeSwappedRAndB, false));
   std::vector<::cv::Mat> outs;
   _net.forward(outs, _net.getUnconnectedOutLayersNames());
   return frameExtract(outs, cv::Size{frame.cols, frame.rows}, std::move(filter));
}

auto YOLOv3::frameExtract(std::vector<::cv::Mat> const& outs, cv::Size const& frameSize, std::function<bool(std::string const&)>&& filter) const -> Item::List
{
    TAKEN_TIME();
   std::vector<int> classIDs;
   std::vector<float> confidences;
   std::vector<::cv::Rect2d> boxes;

   for (const auto& out : outs)
   {
      auto data = out.ptr<float>();
      for (int j = 0; j < out.rows; j++, data += out.cols)
      {
         auto scores = out.row(j).colRange(5, out.cols);
         ::cv::Point classIdPoint;
         double confidence;
         ::cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
         if (confidence > _confThreshold)
         {
            const auto centerX = data[0];
            const auto centerY = data[1];
            const auto width = data[2];
            const auto height = data[3];
            const auto left = centerX - width / 2;
            const auto top = centerY - height / 2;

            classIDs.push_back(classIdPoint.x);
            confidences.push_back(static_cast<float>(confidence));
            boxes.emplace_back(left, top, width, height);
         }
      }
   }
   std::vector<int> indices;
   ::cv::dnn::NMSBoxes(boxes, confidences, _confThreshold, _nmsThreshold, indices);

   Item::List result;
   result.reserve(indices.size());
   for (const auto index : indices)
   {
      if (!filter(_classes[classIDs[index]]))
      {
          continue;
      }
      cv::Rect2f rectInAbsoluteCoords {static_cast<float>(boxes[index].x) * frameSize.width,
                                       static_cast<float>(boxes[index].y) * frameSize.height,
                                       static_cast<float>(boxes[index].width) * frameSize.width,
                                       static_cast<float>(boxes[index].height) * frameSize.height};
      result.emplace_back(Item{_classes[classIDs[index]], confidences[index], rectInAbsoluteCoords});
   }
   return result;
}