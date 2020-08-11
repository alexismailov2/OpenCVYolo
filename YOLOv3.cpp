#include "YOLOv3.hpp"

#include "TimeMeasuring.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/dnn/all_layers.hpp>

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
#if 0

def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    if (float(net_w)/image_w) < (float(net_h)/image_h):
        new_w = net_w
        new_h = (image_h*net_w)/image_w
    else:
        new_h = net_w
        new_w = (image_w*net_h)/image_h

    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h

        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

# decode the output of the network
        for j in range(len(yolos)):
            yolo_anchors = anchors[(2-j)*6:(3-j)*6] # config['model']['anchors']
            boxes += decode_netout(yolos[j], yolo_anchors, obj_thresh, net_h, net_w)

        # correct the sizes of the bounding boxes
        correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5

    boxes = []

    netout[..., :2]  = _sigmoid(netout[..., :2])
    netout[..., 4]   = _sigmoid(netout[..., 4])
    netout[..., 5:]  = netout[..., 4][..., np.newaxis] * _softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h*grid_w):
        row = i // grid_w
        col = i % grid_w

        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[row, col, b, 4]

            if(objectness <= obj_thresh): continue

            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[row,col,b,:4]

            x = (col + x) / grid_w # center position, unit: image width
            y = (row + y) / grid_h # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height

            # last elements are class probabilities
            classes = netout[row,col,b,5:]

            box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)

            boxes.append(box)

    return boxes
#endif

float logistic_activate(float x)
{
   return 1.F / (1.F + exp(-x));
}

void do_nms_sort(float *detections, int total, float score_thresh, float nms_thresh, int classes, int coords = 4)
{
   std::vector<cv::Rect2d> boxes(total);
   std::vector<float> scores(total);

   for (int i = 0; i < total; ++i)
   {
      cv::Rect2d &b = boxes[i];
      int box_index = i * (classes + coords + 1);
      b.width = detections[box_index + 2];
      b.height = detections[box_index + 3];
      b.x = detections[box_index + 0] - b.width / 2;
      b.y = detections[box_index + 1] - b.height / 2;
   }

   std::vector<int> indices;
   for (int k = 0; k < classes; ++k)
   {
      for (int i = 0; i < total; ++i)
      {
         int box_index = i * (classes + coords + 1);
         int class_index = box_index + 5;
         scores[i] = detections[class_index + k];
         detections[class_index + k] = 0;
      }
      cv::dnn::NMSBoxes(boxes, scores, score_thresh, nms_thresh, indices);
      for (int i = 0, n = indices.size(); i < n; ++i)
      {
         int box_index = indices[i] * (classes + coords + 1);
         int class_index = box_index + 5;
         detections[class_index + k] = scores[indices[i]];
      }
   }
}

auto yolov3_decode(cv::Mat netout, std::vector<uint32_t> anchors_list, float threshold, float nmsThreshold, cv::Size netSize) -> cv::Mat
{
   int const anchors = anchors_list.size()/2;
   int const cell_size = netout.size[1]/anchors;
   int const rows = netout.size[2];
   int const cols = netout.size[3];
   int const classes = cell_size - 5;
   int const classfix = 0;

   cv::Mat netoutSized = cv::Mat(cell_size*anchors, rows*cols, CV_32FC1, netout.ptr<float>());
   cv::Mat decoded = cv::Mat::zeros(rows*cols*anchors, cell_size, CV_32FC1);

   // address length for one image in batch, both for input and output
   float*       srcData = netout.ptr<float>();
   float*       dstData = decoded.ptr<float>();
   for (int i = 0; i < rows*cols*anchors; ++i)
   {
      srcData[i * cell_size] = 0.0f;
   }
   std::cout << netoutSized << std::endl;

   // logistic activation for t0, for each grid cell (X x Y x Anchor-index)
   for (int i = 0; i < rows*cols*anchors; ++i)
   {
      int index = cell_size * i;
      float x = srcData[index + 4];
      dstData[index + 4] = logistic_activate(x);	// logistic activation
   }
   for (int i = 0; i < rows*cols*anchors; ++i)
   {
      int index = cell_size * i;
      const float* input = srcData + index + 5;
      float* output = dstData + index + 5;
      for (int c = 0; c < classes; ++c)
      {
         output[c] = logistic_activate(input[c]);
      }
   }
   for (int x = 0; x < cols; ++x)
   {
      for (int y = 0; y < rows; ++y)
      {
         for (int a = 0; a < anchors; ++a)
         {
            // relative start address for image b within the batch data
            int index = (y * cols + x) * anchors + a; // index for each grid-cell & anchor
            int p_index = index * cell_size + 4;
            float scale = dstData[p_index];
            if (classfix == -1 && scale < .5)
            {
               scale = 0; // if(t0 < 0.5) t0 = 0;
            }
            int box_index = index * cell_size;

            auto boxX = (x + logistic_activate(srcData[box_index + 0])) / cols;
            auto boxY = (y + logistic_activate(srcData[box_index + 1])) / rows;
            auto boxW = exp(srcData[box_index + 2]) * anchors_list[2 * a] / netSize.width;
            auto boxH = exp(srcData[box_index + 3]) * anchors_list[2 * a + 1] / netSize.height;

            dstData[box_index + 0] = boxX;
            dstData[box_index + 1] = boxY;
            dstData[box_index + 2] = boxW;
            dstData[box_index + 3] = boxH;

            int class_index = index * cell_size + 5;
            for (int j = 0; j < classes; ++j)
            {
               float prob = scale * dstData[class_index + j];         // prob = IoU(box, object) = t0 * class-probability
               dstData[class_index + j] = (prob > threshold) ? prob : 0; // if (IoU < threshold) IoU = 0;
            }
         }
      }
   }
   if (nmsThreshold > 0)
   {
      do_nms_sort(dstData, rows*cols*anchors, threshold, nmsThreshold, classes);
   }
   return decoded.clone();
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
   _net.setPreferableBackend(::cv::dnn::DNN_BACKEND_CUDA);
   _net.setPreferableTarget(::cv::dnn::DNN_TARGET_CUDA);
}

YOLOv3::YOLOv3(std::string const& modelFile,
               std::string const& classesFile,
               std::vector<std::vector<uint32_t>> const& anchorsList,
               cv::Size inputSize,
               float confThreshold,
               float nmsThreshold)
   : _classes{readClasses(classesFile)}
   , _anchorsList{anchorsList}
   , _inputSize{inputSize}
   , _confThreshold{confThreshold}
   , _nmsThreshold{nmsThreshold}
   , _net{cv::dnn::readNet(modelFile)}
{
   _net.setPreferableBackend(::cv::dnn::DNN_BACKEND_CUDA);
   _net.setPreferableTarget(::cv::dnn::DNN_TARGET_CUDA);
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

   auto i = 2;
   for (const auto& out : outs)
   {
      cv::Mat decoded = (!_anchorsList.empty()) ? yolov3_decode(out, _anchorsList[i--], 0.4, 0.4, _inputSize) : out;
      auto data = decoded.ptr<float>();
      for (int j = 0; j < decoded.rows; j++, data += decoded.cols)
      {
         auto scores = decoded.row(j).colRange(5, decoded.cols);
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