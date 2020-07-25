#pragma once

#include <opencv2/dnn.hpp>

class YOLOv3
{
public:
   struct Item
   {
      using List = std::vector<Item>;

      std::string const& className;
      float              confidence;
      ::cv::Rect2d       boundingBox;
   };

public:
    YOLOv3(std::string const& modelFile,
           std::string const& weightsFile,
           std::string const& classesFile,
           cv::Size inputSize,
           float confidenceThreshold = 0.25f,
           float nmsThreshold = 0.25f);

    auto performPrediction(cv::Mat const& frame,
                           std::function<bool(std::string const&)>&& filter = [](std::string const&) { return true; },
                           bool isNeededToBeSwappedRAndB = true) -> Item::List;

private:
   auto frameExtract(std::vector<::cv::Mat> const& outs,
                     cv::Size const& frameSize,
                     std::function<bool(std::string const&)>&& filter) const -> Item::List;

private:
    std::vector<std::string> _classes;
    cv::Size                 _inputSize;
    float                    _confThreshold;
    float                    _nmsThreshold;
    cv::dnn::Net             _net;
};
