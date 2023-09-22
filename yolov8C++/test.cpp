#include <iostream>
#include <unistd.h>
#include <gst/gst.h>
#include <gst/app/app.h>
// ns
// #define STB_IMAGE_IMPLEMENTATION
// ns

#include <cstdio>
#include <list>
#include <utility>
#include <chrono>

#include <algorithm>
#include <functional>
#include <queue>

#include <cstdarg>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
#include <numeric>
#include <map>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>

#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/optional_debug_tools.h>

#include <tensorflow/lite/delegates/nnapi/nnapi_delegate.h>
#include <tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h>
#include <tensorflow/lite/tools/command_line_flags.h>
#include <tensorflow/lite/tools/delegates/delegate_provider.h>
#include <tensorflow/lite/tools/evaluation/utils.h>

#include <tensorflow/lite/delegates/external/external_delegate.h>

#include "tensorflow/lite/interpreter.h"
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/optional_debug_tools.h>
#include <tensorflow/lite/minimal_logging.h>
#include "vsi_npu_custom_op.h"



#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "utils.hpp"
#include "nms.hpp"

#include <dlfcn.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/tools/logging.h"



using namespace std;
using namespace cv;
using namespace tflite;
// ns
using TfLiteDelegatePtr = tflite::Interpreter::TfLiteDelegatePtr;
// ns
using TfLiteDelegatePtrMap = std::map<std::string, TfLiteDelegatePtr>;

typedef cv::Point3_<float> Pixel;




auto mat_process(cv::Mat src, uint width, uint height) -> cv::Mat
{
  // convert to float; BGR -> RGB
  cv::Mat dst2;
  cout << "Creating dst" << endl;
  // src.convertTo(dst, CV_32FC3);
  cout << "Creating dst2" << endl;
  cv::cvtColor(src, dst2, cv::COLOR_BGR2RGB);
  cout << "Creating dst3" << endl;


  cv::Mat normalizedImage(dst2.rows, dst2.cols, CV_32FC3);

  for (int i = 0; i < dst2.rows; i++) {
    for (int j = 0; j < dst2.cols; j++) {
      cv::Vec3b pixel = dst2.at<cv::Vec3b>(i, j);
      cv::Vec3f normalizedPixel;
      // std::cout << static_cast<float>(pixel[0]) / 255.0f << endl;
      normalizedPixel[0] = static_cast<float>(pixel[0]) / 255.0f;
      normalizedPixel[1] = static_cast<float>(pixel[1]) / 255.0f;
      normalizedPixel[2] = static_cast<float>(pixel[2]) / 255.0f;
      normalizedImage.at<cv::Vec3f>(i, j) = normalizedPixel;
    }
  }

  // cv::Mat normalizedImage(src.rows, src.cols, CV_32FC3);

  // for (int i = 0; i < src.rows; i++) {
  //   for (int j = 0; j < src.cols; j++) {
  //     cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
  //     cv::Vec3f normalizedPixel;
  //     // std::cout << static_cast<float>(pixel[0]) / 255.0f << endl;
  //     normalizedPixel[0] = static_cast<float>(pixel[0]) / 255.0f;
  //     normalizedPixel[1] = static_cast<float>(pixel[1]) / 255.0f;
  //     normalizedPixel[2] = static_cast<float>(pixel[2]) / 255.0f;
  //     normalizedImage.at<cv::Vec3f>(i, j) = normalizedPixel;
  //   }
  // }

  // for (int i = 0; i < normalizedImage.rows; i++) {
  //   for (int j = 0; j < normalizedImage.cols; j++) {
  //       cv::Vec3f normalizedPixel = normalizedImage.at<cv::Vec3f>(i, j);
  //       std::cout << "Pixel at (" << i << ", " << j << "): "
  //                 << "R=" << normalizedPixel[0] << ", "
  //                 << "G=" << normalizedPixel[1] << ", "
  //                 << "B=" << normalizedPixel[2] << std::endl;
  //   }
  // }
  //here is totally same with python
  // cv::resize(normalizedImage, normalizedImage, cv::Size(width, height));

  return normalizedImage;
}


cv::Mat letterbox(cv::Mat img, int height, int width) {
    cv::Size shape = img.size(); // current shape [height, width]
    cv::Size new_shape(640, 640);
    // cv::Size new_shape(10,10);

    // Scale ratio (new / old)
    float r = std::min(static_cast<float>(new_shape.height) / shape.height,
                        static_cast<float>(new_shape.width) / shape.width);

    // Compute padding
    cv::Size new_unpad(static_cast<int>(std::round(shape.width * r)),
                       static_cast<int>(std::round(shape.height * r)));
    float dw = static_cast<float>(new_shape.width - new_unpad.width);
    float dh = static_cast<float>(new_shape.height - new_unpad.height);

    dw /= 2; // divide padding into 2 sides
    dh /= 2;

    if (shape != new_unpad) { // resize
        cv::resize(img, img, new_unpad, 0, 0, cv::INTER_LINEAR);
    }

    int top = static_cast<int>(std::round(dh - 0.1f));
    int bottom = static_cast<int>(std::round(dh + 0.1f));
    int left = static_cast<int>(std::round(dw - 0.1f));
    int right = static_cast<int>(std::round(dw + 0.1f));

    cv::Mat result_img;
    cv::copyMakeBorder(img, result_img, top, bottom, left, right, cv::BORDER_CONSTANT,
                       cv::Scalar(114, 114, 114));

    return result_img;
}






//fprintf(stderr, "minimal <external_delegate.so> <tflite model> <use_cache_mode> <cache file> <inputs>\n");
void setupInput(const std::unique_ptr<tflite::Interpreter>& interpreter) {

   auto in_tensor = interpreter->input_tensor(0);

    switch (in_tensor->type) {
      case kTfLiteFloat32:
      {
        std::cout << "datatype for input kTfLiteFloat32" << std::endl;
        //auto in = ReadData(argv[2], input_data, input_idx, in_tensor->bytes);
        //memcpy(interpreter->typed_input_tensor<float>(input_idx), in.data(), in.size());
        break;
      }
      case kTfLiteUInt8:
      {
        std::cout << "datatype for input kTfLiteUInt8" << std::endl;
        //auto in = ReadData(argv[2], input_data, input_idx, in_tensor->bytes);
        //memcpy(interpreter->typed_input_tensor<uint8_t>(input_idx), in.data(), in.size());
        break;
      }
      case kTfLiteInt8: {
        std::cout << "datatype for input kTfLiteInt8" << std::endl;
        //auto in = ReadData(argv[2], input_data, input_idx, in_tensor->bytes);
        //memcpy(interpreter->typed_input_tensor<int8_t>(input_idx), in.data(), in.size());
        break;
      }
      case kTfLiteInt32:
      {
        std::cout << "datatype for input kTfLiteInt32" << std::endl;
        //auto in = ReadData(argv[2], input_data, input_idx, in_tensor->bytes);
        //memcpy(interpreter->typed_input_tensor<int32_t>(input_idx), in.data(), in.size());
        break;
      }
      default: {
        std::cout << "Fatal: datatype for input not implemented" << std::endl;
        //TFLITE_EXAMPLE_CHECK(false);
        break;
      }
    }

    

}

std::vector<float> xywh2xyxy_scale(const std::vector<float>& boxes, float width, float height) {
    std::vector<float> result;
    for (size_t i = 0; i < boxes.size(); i += 4) {
        float x = boxes[i];
        float y = boxes[i + 1];
        float w = boxes[i + 2];
        float h = boxes[i + 3];

        float x1 = (x - w / 2) * width;     // top left x
        float y1 = (y - h / 2) * height;    // top left y
        float x2 = (x + w / 2) * width;     // bottom right x
        float y2 = (y + h / 2) * height;    // bottom right y

        result.push_back(x1);
        result.push_back(y1);
        result.push_back(x2);
        result.push_back(y2);
    }
    return result;
}

// //doesnt make a differentce
// template<typename T>
// auto cvtTensor(TfLiteTensor* tensor) -> vector<T>;

// auto cvtTensor(TfLiteTensor* tensor) -> vector<float>{
//     int nelem = 1;
//     for(int i=0; i<tensor->dims->size; ++i)
//         nelem *= tensor->dims->data[i];
//     vector<float> data(tensor->data.f, tensor->data.f+nelem);
//     return data;
// }


// std::vector<float> scaleBox(const std::vector<float>& box, int img1Height, int img1Width, int img0Height, int img0Width) {
//     std::vector<float> scaledBox = box;

//     // Calculate gain and padding
//     float gain = std::min(static_cast<float>(img1Height) / img0Height, static_cast<float>(img1Width) / img0Width);
//     int padX = static_cast<int>((img1Width - img0Width * gain) / 2 - 0.1);
//     int padY = static_cast<int>((img1Height - img0Height * gain) / 2 - 0.1);

//     // Apply padding and scaling
//     scaledBox[0] -= padX;
//     scaledBox[2] -= padX;
//     scaledBox[1] -= padY;
//     scaledBox[3] -= padY;

//     scaledBox[0] /= gain;
//     scaledBox[2] /= gain;
//     scaledBox[1] /= gain;
//     scaledBox[3] /= gain;

//     // Clip the box
//     scaledBox[0] = std::max(0.0f, std::min(scaledBox[0], static_cast<float>(img0Width)));
//     scaledBox[2] = std::max(0.0f, std::min(scaledBox[2], static_cast<float>(img0Width)));
//     scaledBox[1] = std::max(0.0f, std::min(scaledBox[1], static_cast<float>(img0Height)));
//     scaledBox[3] = std::max(0.0f, std::min(scaledBox[3], static_cast<float>(img0Height)));

//     return scaledBox;
// }


// std::vector<int> NMS(const std::vector<std::vector<float>>& boxes, float overlapThresh) {
//     // Return an empty vector if no boxes given
//     if (boxes.empty()) {
//         return std::vector<int>();
//     }

//     std::vector<float> x1, y1, x2, y2, areas;
//     std::vector<int> indices(boxes.size());

//     // Extract coordinates and compute areas
//     for (size_t i = 0; i < boxes.size(); ++i) {
//         x1.push_back(boxes[i][0]);
//         y1.push_back(boxes[i][1]);
//         x2.push_back(boxes[i][2]);
//         y2.push_back(boxes[i][3]);
//         areas.push_back((x2[i] - x1[i] + 1) * (y2[i] - y1[i] + 1));
//         indices[i] = static_cast<int>(i);
//     }

//     std::vector<int> keep;

//     for (size_t i = 0; i < boxes.size(); ++i) {
//         std::vector<int> tempIndices;
//         for (size_t j = 0; j < boxes.size(); ++j) {
//             if (j != i) {
//                 tempIndices.push_back(static_cast<int>(j));
//             }
//         }

//         float xx1 = std::max(x1[i], x1[tempIndices[0]]);
//         float yy1 = std::max(y1[i], y1[tempIndices[0]]);
//         float xx2 = std::min(x2[i], x2[tempIndices[0]]);
//         float yy2 = std::min(y2[i], y2[tempIndices[0]]);

//         float w = std::max(0.0f, xx2 - xx1 + 1);
//         float h = std::max(0.0f, yy2 - yy1 + 1);

//         float overlap = (w * h) / areas[tempIndices[0]];

//         if (overlap <= overlapThresh) {
//           for (int index : indices) {
//               if (index != i) {
//                   indices.push_back(index);
//               }
//           }
//           keep.push_back(indices[i]);
//         }
//     }

//     return keep;
// }


// std::vector<int> NMS(std::vector<std::vector<float>>& boxes, float overlapThreshold) {
//     // Return an empty list, if no boxes given
//     if (boxes.empty()) {
//         return std::vector<int>();
//     }

//     std::vector<float> x1, y1, x2, y2, areas;
//     std::vector<int> indices;

//     // Extract x1, y1, x2, y2 from the boxes
//     for (const auto& box : boxes) {
//         x1.push_back(box[0]);
//         y1.push_back(box[1]);
//         x2.push_back(box[2]);
//         y2.push_back(box[3]);
//     }

//     // Compute the area of the bounding boxes and sort the bounding
//     // Boxes by the bottom-right y-coordinate of the bounding box
//     for (size_t i = 0; i < boxes.size(); ++i) {
//         float area = (x2[i] - x1[i] + 1) * (y2[i] - y1[i] + 1);
//         areas.push_back(area);
//         indices.push_back(static_cast<int>(i));
//     }

//     for (size_t i = 0; i < boxes.size(); ++i) {
//         std::vector<int> temp_indices;
//         for (size_t j = 0; j < indices.size(); ++j) {
//             if (indices[j] != static_cast<int>(i)) {
//                 temp_indices.push_back(indices[j]);
//             }
//         }

//         float xx1 = std::max(boxes[i][0], boxes[temp_indices[0]][0]);
//         float yy1 = std::max(boxes[i][1], boxes[temp_indices[0]][1]);
//         float xx2 = std::min(boxes[i][2], boxes[temp_indices[0]][2]);
//         float yy2 = std::min(boxes[i][3], boxes[temp_indices[0]][3]);

//         float w = std::max(0.0f, xx2 - xx1 + 1);
//         float h = std::max(0.0f, yy2 - yy1 + 1);

//         float overlap = (w * h) / areas[temp_indices[0]];

//         if (overlap <= overlapThreshold) {
//             // indices.erase(indices.begin() + i);

//         }
//     }

//     return indices;
// }




std::vector<std::vector<float>> process_4(const std::unique_ptr<tflite::Interpreter>& interpreter,string infile, string outfile, vector<Rect> &nmsrec, vector<int> &pick, vector<int> &ids, cv::Mat *inp = NULL)
{


  //setupInput(interpreter);

  cout << " Got model " << endl;
  // get input & output layer
  TfLiteTensor *input_tensor = interpreter->tensor(interpreter->inputs()[0]);
  cout << " Got input " << endl;
  TfLiteTensor *output_box = interpreter->tensor(interpreter->outputs()[0]);
  cout << " Got output " << endl;
  // TfLiteTensor *output_score = interpreter->tensor(interpreter->outputs()[1]);
  // cout << " Got output score " << endl;

  const uint HEIGHT = input_tensor->dims->data[1];
  const uint WIDTH = input_tensor->dims->data[2];
  const uint CHANNEL = input_tensor->dims->data[3];
  // cout << "H " << HEIGHT << " W " << WIDTH << " C " << CHANNEL << endl;

  // read image file
  std::chrono::time_point<std::chrono::system_clock> beg, start, end, done, nmsdone;
  std::chrono::duration<double> elapsed_seconds;
  start = std::chrono::system_clock::now();

  cv::Mat img;
  if (inp == NULL)
  {
    cout << "Getting image from file " << endl;
    img = cv::imread(infile);
  }
  else
  {
    cout << "Getting image from input " << endl;
    img = *inp;
  }

  const float width=img.rows;
  const float height=img.cols;

  end = std::chrono::system_clock::now();
  elapsed_seconds = end - start;
  printf("Read matrix from file s: %.10f\n", elapsed_seconds.count());

  start = std::chrono::system_clock::now();

  // img = mat_process(img, WIDTH, HEIGHT); // could be input is modified by this function



  //here is the same with python
  cv::Mat inputImg = letterbox(img, WIDTH, HEIGHT);


  // for (int y = 0; y < inputImg.rows; y++) {
  //     for (int x = 0; x < inputImg.cols; x++) {
  //         // Get the RGB values at the current pixel
  //         cv::Vec3b pixel = inputImg.at<cv::Vec3b>(y, x);
  //         int blue = pixel[0];
  //         int green = pixel[1];
  //         int red = pixel[2];

  //         // Print the RGB values for the current pixel
  //         std::cout << "Pixel at (" << x << ", " << y << "): R=" << red << " G=" << green << " B=" << blue << std::endl;
  //     }
  // }


  inputImg = mat_process(inputImg, WIDTH, HEIGHT);


  // for (int y = 0; y < inputImg.rows; y++) {
  //     for (int x = 0; x < inputImg.cols; x++) {
  //         // Get the RGB values at the current pixel
  //         cv::Vec3f pixel = inputImg.at<cv::Vec3f>(y, x);
  //         float blue = pixel[0];
  //         float green = pixel[1];
  //         float red = pixel[2];

  //         // Print the RGB values for the current pixel
  //         std::cout << "Pixel at (" << x << ", " << y << "): R=" << red << " G=" << green << " B=" << blue << std::endl;
  //     }
  // }



  cout << "DIM IS " << inputImg.channels() << endl;


  cout << " Got image " << endl;

  end = std::chrono::system_clock::now();
  elapsed_seconds = end - start;
  printf("Process Matrix to RGB s: %.10f\n", elapsed_seconds.count());
  interpreter->SetAllowFp16PrecisionForFp32(true);

  start = std::chrono::system_clock::now();
  // cout << " GOT INPUT IMAGE " << endl;
  
  // flatten rgb image to input layer.
  // float* input_data = interpreter->typed_input_tensor<float>(0);
  memcpy(input_tensor->data.f, inputImg.ptr<float>(0), 640*640*3* sizeof(float));

  // cout << " the 600001 is " << input_tensor->data.f[600001] << endl;
  // cout << " the 600002 is " << input_tensor->data.f[600002] << endl;
  // cout << " the 600003 is " << input_tensor->data.f[600003] << endl;
  // cout << " the 600004 is " << input_tensor->data.f[600004] << endl;
  // cout << " the 600005 is " << input_tensor->data.f[600005] << endl;
  // cout << " the 600006 is " << input_tensor->data.f[600006] << endl;

  // float *inputImg_ptr = inputImg.ptr<float>(0);
  // memcpy(input_tensor->data.f, inputImg.ptr<float>(0),
  //        WIDTH * HEIGHT * CHANNEL * sizeof(float));

  // for (int i = 0; i < WIDTH * HEIGHT * CHANNEL; i++) {
  //     float value = input_tensor->data.f[i];
  //     cout << "Value at index " << i << ": " << value << endl;
  // }
  

  // cout << " GOT MEMCPY " << endl;
  // compute model instance

  interpreter->Invoke();
  end = std::chrono::system_clock::now();
  elapsed_seconds = end - start;
  printf("invoke interpreter s: %.10f\n", elapsed_seconds.count());

  float *box_vec = interpreter->typed_output_tensor<float>(0);
  // float* output2 = interpreter->typed_output_tensor<float>(1);
  // printf("%f ", output2[0]);

  // vector<float> box_vec = cvtTensor(output_box);


  vector<cv::Rect> rects;
  vector<vector<float>> recvec;
  vector<float> scores;
  // vector<int> ids;
  vector<int> nms;
  map<int, vector<vector<float>>> map_rects;
  map<int, vector<float>> map_scores;

  float max_class_score = 0;
  int max_class = -1;
  float MAX_ACCEPT_SCORE = 0.25;
  float bbox_score = 1;
  float tindex = -1;

  int nelem = 1;
  int dim1 = output_box->dims->data[2]; // should be 8400
  int dim2 = output_box->dims->data[1]; // should be 84
  for (int i = 0; i < output_box->dims->size; ++i)
  {
    cout << "DIM IS " << output_box->dims->data[i] << endl;
    nelem *= output_box->dims->data[i];
  }
  //output is 1x84x8400
  // cout << " bounding boxes " << endl;

  // cout << " first line " << endl;

  // printf("%f ", box_vec[0]);
  // printf("%f ", box_vec[1]);
  // printf("%f ", box_vec[2]);
  // printf("%f ", box_vec[3]);
  // printf("%f ", box_vec[1*8400-1]);
  // printf("%f ", box_vec[1*8400-2]);
  // printf("%f ", box_vec[1*8400-3]);
  // printf("%f ", box_vec[1*8400-4]);

  // cout << " first col " << endl;
  // printf("%f ", box_vec[0]);
  // printf("%f ", box_vec[1*8400+1]);
  // printf("%f ", box_vec[1*8400+2]);
  // printf("%f ", box_vec[1*8400+3]);
  // printf("%f ", box_vec[2*8400-1]);
  // printf("%f ", box_vec[2*8400-2]);
  // printf("%f ", box_vec[2*8400-3]);
  // printf("%f ", box_vec[2*8400-4]);
  // //now here is correct


  //same way with python

  // std::vector<std::vector<float>> boxes;
  // std::vector<std::vector<float>> probs;

  //ready to delete
  // cv::Mat confidence;

  // cv::Mat bbox;
  // int base = 4 * 8400;
  // int m = 0;


  // while (m < 8400) {
  //   std::vector<float> temp;
  //   int n = 0;

  //   std::vector<float> temp1;

  //   while (n < 80) {

 
  //     temp1.push_back(box_vec[n * 8400 + m + base]);


  //     std::vector<float> temp2;
  //     int i = 0;
  //     while (i < 4) {
  //       temp2.push_back(box_vec[i * 8400 + m]);
  //       i++;
  //     }


  //     // bbox.push_back(xywh2xyxy_scale(temp2,640,640));
  //     bbox.push_back(temp2);

  //     n++;
  //   }
  //   confidence.push_back(temp1);
  //   m++;
  // }

  // std::cout << confidence.rows << confidence.cols<< std::endl;
  // std::cout << bbox.rows << bbox.cols<< std::endl;  
  //ready to delete



  //use this
  std::vector<float> confidence;
  std::vector<float> index;
  std::vector<std::vector<float>> bbox;



  int base = 4 * 8400;
  int m = 0;


  while (m < 8400) {
    std::vector<float> temp;
    int n = 0;


    while (n < 80) {
      if (box_vec[n * 8400 + m + base] > 0.25) {
        index.push_back(n);
        confidence.push_back(box_vec[n * 8400 + m + base]);


        std::vector<float> temp2;
        int i = 0;
          while (i < 4) {
            temp2.push_back(box_vec[i * 8400 + m]);
            i++;
          }


          // bbox.push_back(xywh2xyxy_scale(temp2,640,640));
          bbox.push_back(temp2);
      }
      n++;
    }
    m++;
  }

  // Print confidence vector
  std::cout << "Confidence vector:" << std::endl;
  for (float val : confidence) {
  std::cout << val << " ";
  }
  std::cout << std::endl;


  // Print index vector
  std::cout << "Index vector:" << std::endl;
  for (float val : index) {
  std::cout << val << " ";
  }
  std::cout << std::endl;


  // Print bbox vector
  std::cout << "Bbox vector:" << std::endl;
  for (const std::vector<float>& box : bbox) {
  for (float val : box) {
  std::cout << val << " ";
  }
  std::cout << std::endl;
  }


  //image0frame40814.jpg
  //Confidence vector:
  // 0.876689 0.345122 0.876689 0.932226 0.904457 0.904457 0.932226 0.904457 0.654542 0.932226 0.932226 
  // Index vector:
  // 0 0 0 0 0 0 0 0 0 0 0 
  // Bbox vector:
  // 0.642641 0.531567 0.170577 0.468096 
  // 0.642641 0.531567 0.170577 0.468096 
  // 0.642641 0.547435 0.170577 0.468096 
  // 0.642641 0.547435 0.170577 0.468096 
  // 0.642641 0.547435 0.170577 0.468096 
  // 0.642641 0.531567 0.170577 0.483964 
  // 0.642641 0.547435 0.170577 0.468096 
  // 0.642641 0.547435 0.186445 0.468096 
  // 0.642641 0.531567 0.170577 0.468096 
  // 0.642641 0.547435 0.170577 0.468096 
  // 0.642641 0.531567 0.186445 0.483964 



  //8880.jpg

  //   Confidence vector:
  // 0.876689 0.345122 0.876689 0.932226 0.904457 0.904457 0.932226 0.904457 0.654542 0.932226 0.932226 
  // Index vector:
  // 0 0 0 0 0 0 0 0 0 0 0 
  // Bbox vector:
  // 0.642641 0.531567 0.170577 0.468096 
  // 0.642641 0.531567 0.170577 0.468096 
  // 0.642641 0.547435 0.170577 0.468096 
  // 0.642641 0.547435 0.170577 0.468096 
  // 0.642641 0.547435 0.170577 0.468096 
  // 0.642641 0.531567 0.170577 0.483964 
  // 0.642641 0.547435 0.170577 0.468096 
  // 0.642641 0.547435 0.186445 0.468096 
  // 0.642641 0.531567 0.170577 0.468096 
  // 0.642641 0.547435 0.170577 0.468096 
  // 0.642641 0.531567 0.186445 0.483964 



  std::vector<std::vector<float>> results;

  // int confidence_length=confidence.size();
  // int class_length=index.size();
  // assert(confidence_length == class_length);

  // for(int i=0;i< confidence_length;i++){
  //   std::vector<int> temp;
  //   for (int j=0;j< class_length;j++){
  //     std::vector<std::vector<float>>box;
  //     if(index[j]==i){
  //       temp.push_back(j);
  //       box.push_back(bbox[j]);
  //     }
      
  //     std::vector<std::vector<float>> box_selected;
  //     std::vector<float> confidence_selected;
  //     if (temp.size()>0){
  //       //what type does nms return
  //       std::vector<int> indices=NMS(box, 0.45); //this line start error:
        // if(indices.size()>0){
        //   for(int k=0;k<indices.size();k++){
        //     box_selected.push_back(box[indices[k]]);
        //     confidence_selected.push_back(confidence[indices[k]]);
        //   }
        //   for(int p=0;p<box_selected.size();p++){
        //     box_selected[p]=scaleBox(box_selected[p], HEIGHT, WIDTH, height, width);
        //   }

        // }
      // }
      // std::vector<float> temp3;
      // for(int u=0;u<box_selected.size();u++){
      //   temp3=box_selected[u];
      //   temp3.push_back(confidence_selected[u]);
      //   float floatValue = static_cast<float>(j);
      //   temp3.push_back(floatValue);
      // }
      // results.push_back(temp3);
  //   }
  // }

  // for (const std::vector<float>& row : results) {
  //   // Iterate through the elements in each row (inner vector)
  //   for (float element : row) {
  //       std::cout << element << ' ';
  //   }
  //   std::cout << std::endl; // Print a newline after each row
  // }



  // //img is the image
  // int size_threshold=3872;
  // std::vector<float> final_result;

  // int confidence_length=confidence.size();
  // int class_length=index.size();
  // assert(confidence_length == class_length);

  


  // for(int i=0; i<confidence_length;i++){
  //   std::vector<float> temp;
  //   cv::Mat detected=img(cv::Rect(bbox[i][0], bbox[i][1], bbox[i][2] - bbox[i][0], bbox[i][3] - bbox[i][1]));
  //   if(detected.rows*detected.cols >= size_threshold){
  //     temp.push_back(bbox[i][0]);
  //     temp.push_back(bbox[i][1]);
  //     temp.push_back(bbox[i][2]);
  //     temp.push_back(bbox[i][3]);
  //     temp.push_back(confidence[i]);
  //     temp.push_back(index[i]);

  //   }
  //   results.push_back(temp);
  // }

  // for (const std::vector<float>& row :results) {
  //   // Iterate through the elements in each row (inner vector)
  //   for (float element : row) {
  //       std::cout << element << ' ';
  //   }
  //   std::cout << std::endl; // Print a newline after each row
  // }

  return results;

}



char *read_image(string image_name, int &filesize)
{
  cout << "Reading image " << endl;
  std::ifstream file(image_name, ios::binary);
  file.seekg(0, std::ios::end);
  filesize = (int)file.tellg();
  file.seekg(0);
  char *output = new char[filesize];
  file.read(output, filesize);
  cout << "IMAGE SIZE IS " << filesize << endl;
  return output;
}

cv::Mat convert_image(char *img, int height, int width, int filesize)
{

  // cv::Mat raw_data;

  // raw_data = cv::imdecode(cv::Mat(1, filesize, CV_8UC1, img), IMREAD_UNCHANGED);
  // cv::Mat raw_data(1,filesize,CV_8UC3,img,cv::Mat::AUTO_STEP);
  // cout << " Got raw data " << raw_data.cols << " rows " << raw_data.rows << endl;
  // cv::Mat result = cv::imdecode(raw_data,IMREAD_UNCHANGED);
  // cout << "Decoded the matrix " << decoded_mat.cols << " rows are " << decoded_mat.rows << endl;
  // return decoded_mat;
  // return result;
  cv::Mat mat_img;

  mat_img = cv::imdecode(cv::Mat(1, filesize, CV_8UC1, img), IMREAD_UNCHANGED);
  return mat_img;
}

// /*cv::Mat get_mat(char* jpeg_img){
//   cv::Mat rawData(1080, 1920, CV_8FC3, (void*)jpeg_img);
//   //cv::Mat inputImg = imdecode(rawData, IMREAD_ANYDEPTH);
//   cout << "Got matrix and converted it to 1920 1080 CV_8SC3..." << endl;
//   return rawData;
// }*/

int main(int argc, char **argv)
{

    // create model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile("yolov8n_integer_quant.tflite");
  
  //auto ext_delegate_option = TfLiteExternalDelegateOptionsDefault("/usr/lib/libvx_delegate.so");
  //auto ext_delegate_ptr = TfLiteExternalDelegateCreate(&ext_delegate_option);
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  //interpreter->ModifyGraphWithDelegate(ext_delegate_ptr);
  interpreter->SetAllowFp16PrecisionForFp32(true);
  interpreter->AllocateTensors();

  std::cout << " Tensorflow Test " << endl;

  // int height =320;
  // int width = 320;
  // string file_name = "image0frame0.jpg";
  // const char* img = read_image(file_name);
  // cout << "Got image char array " << endl;
  // string resp = process_frame(img, height, width);

  string imgf1= "image0frame40814.jpg";
  // string imgf1 ="8880.jpg";
  string imgf2 = "image0frame30.jpg";
  if (argc == 3)
  {
    imgf1 = argv[1];
    imgf2 = argv[2];
    cout << imgf1 << " " << imgf2 << endl;
  }

  vector<Rect> nmsrec1, nmsrec2, nmsrec3;
  vector<int> pick1, pick2, pick3;
  vector<int> ids1, ids2, ids3;
  vector<int> motion1, motion2, motion3;

  int filesize = 0;
  char *img1 = read_image(imgf1, filesize);
  cv::Mat img1_mat = convert_image(img1, 1072, 1920, filesize);
  // char *img2 = read_image(imgf2, filesize);
  // cv::Mat img2_mat = convert_image(img2, 1080, 1920, filesize);
  // cv::Mat img1 = cv::imread("./0030.jpg");
  // cv::Mat img1 = cv::imread("./image0frame0.jpg");
  // cv::Mat img2 = cv::imread("./image0frame30.jpg");
  // if (!img1.data) {
  //   std::cerr << "Error: Could not open or read the image file." << std::endl;
  //   return -1;
  // }

  // cv::Mat img1_mat = cv::imread("./009000.jpg");  

  std::vector<std::vector<float>> pmat1 = process_4(interpreter,"", "./result0.jpg", nmsrec1, pick1, ids1,  &img1_mat);

  // cv::Mat pmat2 = process_4(interpreter,"", "./result1.jpg", nmsrec2, pick2, ids2,  &img2);
  


  return 0;


}