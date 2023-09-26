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



std::vector<float> scaleBox(const std::vector<float>& box, int img1Height, int img1Width, int img0Height, int img0Width) {
    std::vector<float> scaledBox = box;

    // Calculate gain and padding
    float gain = std::min(static_cast<float>(img1Height) / img0Height, static_cast<float>(img1Width) / img0Width);
    int padX = static_cast<int>((img1Width - img0Width * gain) / 2 - 0.1);
    int padY = static_cast<int>((img1Height - img0Height * gain) / 2 - 0.1);

    // Apply padding and scaling
    scaledBox[0] -= padX;
    scaledBox[2] -= padX;
    scaledBox[1] -= padY;
    scaledBox[3] -= padY;

    scaledBox[0] /= gain;
    scaledBox[2] /= gain;
    scaledBox[1] /= gain;
    scaledBox[3] /= gain;

    // Clip the box
    scaledBox[0] = std::max(0.0f, std::min(scaledBox[0], static_cast<float>(img0Width)));
    scaledBox[2] = std::max(0.0f, std::min(scaledBox[2], static_cast<float>(img0Width)));
    scaledBox[1] = std::max(0.0f, std::min(scaledBox[1], static_cast<float>(img0Height)));
    scaledBox[3] = std::max(0.0f, std::min(scaledBox[3], static_cast<float>(img0Height)));

    return scaledBox;
}



std::vector<int> NMS(const std::vector<std::vector<float>>& boxes, float overlapThresh) {
    // Return an empty vector if no boxes given
    if (boxes.empty()) {
        return std::vector<int>();
    }
    // std::cout << "================" << std::endl;
    // // std::cout << boxes.size() << std::endl;
    // for (const std::vector<float>& box : boxes) {
    //     for (const float& value : box) {
    //         std::cout << value << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "================" << std::endl;
    // std::cout << boxes.size() << boxes[0].size() << std::endl;
    std::vector<float> x1, y1, x2, y2, areas;
    std::vector<int> indices;



    // Extract coordinates and compute areas

    int a =0;
    while(a<boxes.size()){
        x1.push_back(boxes[a][0]);    
        y1.push_back(boxes[a][1]);
        x2.push_back(boxes[a][2]);
        y2.push_back(boxes[a][3]);
        areas.push_back((x2[a] - x1[a] + 1) * (y2[a] - y1[a] + 1));
        indices.push_back(a);
        a++;
    }
    // std::cout << areas.size() <<std::endl;
    // for(int o=0;o<areas.size();o++){
    //   std::cout << areas[o]  << "pppp" <<std::endl;
    // }



    // std::cout << boxes.size() << std::endl;



    for(int q=0; q<boxes.size();q++){
      
      std::vector<int>temp_indices;
      for(int p=0;p<indices.size();p++){
        if(indices[p]!=q){
          temp_indices.push_back(indices[p]);
        }
      }
      //q and temp_indices

      vector<float> xx1;
      vector<float> yy1;
      vector<float> xx2;
      vector<float> yy2;


      for(int l=0;l<temp_indices.size();l++){
        xx1.push_back(std::max(boxes[temp_indices[l]][0], boxes[q][0]));
        yy1.push_back(std::max(boxes[temp_indices[l]][1], boxes[q][1]));
        xx2.push_back(std::min(boxes[temp_indices[l]][2], boxes[q][2]));
        yy2.push_back(std::min(boxes[temp_indices[l]][3], boxes[q][3]));
      }

      // for (const float& value : xx1) {
      //   std::cout << value << " ";
      // }
      // std::cout << "====="<< std::endl;
      // for (const float& value : xx2) {
      //   std::cout << value << " ";
      // }
      // std::cout << "====="<< std::endl;

      assert( xx2.size() == xx1.size());
      
      vector<float>w;
      for(int x=0; x< xx1.size();x++){

        w.push_back(std::max(0.0f,(xx2[x]-xx1[x]+1)));
        // std::cout << xx2[x] << xx1[x] <<std::endl;

      }

      assert( yy2.size() == yy1.size());

      vector<float>h;
      for(int y=0; y<yy1.size();y++){
          h.push_back(std::max(0.0f, (yy2[y]-yy1[y]+1)));
      }

      vector<float> temp_areas;
      for(int l=0;l<temp_indices.size();l++){
        temp_areas.push_back(areas[temp_indices[l]]);
      }
      // for(int v=0;v<temp_areas.size();v++){
      //     std::cout << temp_areas[v] << std::endl;
      // }

      vector<float> wxh;
      assert(w.size()==h.size());
      for(int b=0;b<w.size();b++){
          wxh.push_back(w[b]*h[b]);
          // std::cout << w[b] << h[b] << std::endl;
          // std::cout << w[b]*h[b] <<std::endl;
      }

      vector<float> overlap;
      assert(wxh.size()==temp_areas.size());
      for(int n=0;n<wxh.size();n++){
          overlap.push_back(wxh[n]/temp_areas[n]);
          // std::cout<< wxh[n] << std::endl;
          // std::cout<< temp_areas[n] << std::endl;
          // std::cout<< wxh[n]/temp_areas[n] << std::endl;
      }
      bool exist=false;

      for (int u=0;u<overlap.size();u++){
        if(overlap[u]>overlapThresh){
          exist=true;
        }
      }


      if(exist){
        vector<int>temp5;
        for(int w=0;w<indices.size();w++){
          if(indices[w]!=q){
            temp5.push_back(indices[w]);
          }
        }
        indices=temp5;
      }

    }
   
    // std::cout << "====" << std::endl;
    // for(int t=0;t<indices.size();t++){
    //   std::cout << indices[t] <<std::endl;
    // }
    // std::cout << "====" << std::endl;

    return indices;
}




std::vector<std::vector<float>> process_4(const std::unique_ptr<tflite::Interpreter>& interpreter,const string& infile, string outfile, vector<Rect> &nmsrec, vector<int> &pick, vector<int> &ids, cv::Mat *inp = NULL)
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
  // std::cout << width << height <<std::endl;
  //width is 1080, height is 1920

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
  memcpy(input_tensor->data.f, inputImg.ptr<float>(0), HEIGHT*WIDTH*3* sizeof(float));

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
  // // now here is correct


  //implemented a faster way in C++

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
      if (box_vec[n * 8400 + m + base] >= 0.25) {
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

  // std::cout << confidence.size()<< std::endl;
  // std::cout << index.size()<< std::endl;
  // std::cout << bbox.size()<< std::endl;



  for(int q=0;q<bbox.size();q++){
      bbox[q]=xywh2xyxy_scale(bbox[q],640,640);
      // for(int z=0;z<bbox[q].size();z++){
      //     // std::cout<< bbox[q][z] << std::endl;

      // }
      // std::cout<< "=======" << std::endl;
  }

  std::vector<std::vector<int>> ind;
  for(int i=0;i< 80;i++){
    std::vector<int> temp4;
    for(int j=0;j<index.size();j++){
      if(i==index[j]){
        // std::cout <<index[j] <<std::endl;
        temp4.push_back(j);
      }
    }

    if(temp4.empty()){
      temp4.push_back(8401);
    }
    ind.push_back(temp4);
    // std::cout << temp4.size() << std::endl;

  }

  // //here i have ind, confidence, index, bbox
  // // Print confidence vector
  // std::cout << "Confidence vector:" << std::endl;
  // for (float val : confidence) {
  // std::cout << val << " ";
  // }
  // std::cout << std::endl;


  // // Print index vector
  // std::cout << "Index vector:" << std::endl;
  // for (float val : index) {
  // std::cout << val << " ";
  // }
  // std::cout << std::endl;


  // // Print bbox vector
  // std::cout << "Bbox vector:" << std::endl;
  // for (const std::vector<float>& box : bbox) {
  // for (float val : box) {
  // std::cout << val << " ";
  // }
  // std::cout << std::endl;
  // }



  //here i have ind, confidence, index, bbox
  std::vector<std::vector<float>> results;

  int confidence_length=confidence.size();
  int class_length=index.size();
  assert(confidence_length == class_length);

  assert(confidence_length == bbox.size());

  std::vector<std::vector<float>> temp_results;

  for (int i=0; i< 80; i++){

      std::vector<std::vector<float>>box_selected;


      std::vector<std::vector<float>> box_afternms;
      // std::vector<float> confidence_selected;
      
      if (ind[i][0]!=8401){
        for(int j=0;j<ind[i].size();j++){
          box_selected.push_back(bbox[ind[i][j]]);
          
        }

        // std::cout << i<< "====" << std::endl;
        // for(int t=0;t<indices.size();t++){
        //   std::cout << indices[t] <<std::endl;
        // }
        // std::cout << "====" << std::endl;
        //nms is totally correct now

        std::vector<int> indices=NMS(box_selected, 0.45);
        if(indices.size()>0){
          for(int s=0;s<indices.size();s++){
            box_afternms.push_back(bbox[ind[i][indices[s]]]);
          }
        }
        // std::cout << "================" << std::endl;
        // // std::cout << boxes.size() << std::endl;
        // for (const std::vector<float>& box : box_afternms) {
        //     for (const float& value : box) {
        //         std::cout << value << " ";
        //     }
        //     std::cout << std::endl;
        // }
        // std::cout << "================" << std::endl;
        // assert(0==1);

        for(int d=0;d<box_afternms.size();d++){
          box_afternms[d]=scaleBox(box_afternms[d], HEIGHT, WIDTH, static_cast<int>(width), static_cast<int>(height) );
        }

        vector<float> confidence_afternms;
        if(indices.size()>0){
          for(int s=0;s<indices.size();s++){
            confidence_afternms.push_back(confidence[ind[i][indices[s]]]);
          }
        }

        assert(box_afternms.size()==confidence_afternms.size());

        
        // std::cout << box_afternms.size() << std::endl;
        for(int f=0;f<box_afternms.size();f++){
          vector<float> temp6;
          temp6.push_back(box_afternms[f][0]);
          temp6.push_back(box_afternms[f][1]);
          temp6.push_back(box_afternms[f][2]);
          temp6.push_back(box_afternms[f][3]);
          temp6.push_back(confidence_afternms[f]);
          temp6.push_back(static_cast<float>(i));
          temp_results.push_back(temp6);
        }

      }
      
  }//end of forloop


  // for (const std::vector<float>& row :temp_results) {
  //   // Iterate through the elements in each row (inner vector)
  //   for (float element : row) {
  //       std::cout << element << ' ';
  //   }
  //   std::cout << std::endl; // Print a newline after each row
  // }

  //aobe is totally correct

  //img is the image
  int size_threshold=3872;
  // std::cout << temp_results.size() << std::endl;
  //compare results in temp_results with threshold
  for(int i=0; i<temp_results.size();i++){
    std::vector<float> temp7;
    std::vector<int> box7;
    box7.push_back(std::round(temp_results[i][0]));
    box7.push_back(std::round(temp_results[i][1]));
    box7.push_back(std::round(temp_results[i][2]));
    box7.push_back(std::round(temp_results[i][3]));
    cv::Mat detected=img(cv::Rect(box7[0], box7[1], box7[2] - box7[0], box7[3] - box7[1]));
    // std::cout << detected.rows*detected.cols << std::endl;
    if(detected.rows*detected.cols >= size_threshold){
      temp7.push_back(temp_results[i][0]);
      temp7.push_back(temp_results[i][1]);
      temp7.push_back(temp_results[i][2]);
      temp7.push_back(temp_results[i][3]);
      temp7.push_back(temp_results[i][4]);
      temp7.push_back(temp_results[i][5]);
      results.push_back(temp7);
    }
    
  }

  for (const std::vector<float>& row :results) {
    // Iterate through the elements in each row (inner vector)
    for (float element : row) {
        std::cout << element << ' ';
    }
    std::cout << std::endl; // Print a newline after each row
  }

  return results;

}



  //  //approach two:
  // cv::Mat boxes(8400,4,CV_32F);
  // cv::Mat probs(8400,80, CV_32F);

  // for(int i=0;i<8400;i++){
  //   for(int j=0;j<4;j++){
  //     boxes.at<float>(i,j)+=box_vec[j*8400+i];
  //   }
  // }

  // for(int i=0;i<8400;i++){
  //   for(int j=4;j<84;j++){
  //      probs.at<float>(i,j)+=box_vec[j*8400+i];
  //   }
  // }
  // cv::Mat scores2(1,8400,CV_32F);
  // cv::Mat index(1,8400,CV_32F);
  // for(int i=0;i<8400;i++){
    

  //   // float max_row=probs.at<float>(i,0);
  //   float max_row=0.0f;
  //   for(int j=0;j<80;j++){

  //     if(probs.at<float>(i,j) > max_row){
  //       scores2.at<float>(1,i)=probs.at<float>(i,j);

  //       index.at<float>(1,i)=static_cast<float>(j);
  //       max_row=probs.at<float>(i,j);
  //     }
  //   }
  // }

  // std::vector<int>idx_th;
  // for(int i=0;i<scores2.cols;i++){
  //   if(scores2.at<float>(1,i)>0.25){
  //     idx_th.push_back(i);
  //   }
  // }

  // //good from here
  // for(int i=0;i<idx_th.size();i++){
  //   std::cout<<idx_th[i]<<std::endl;
  // }

  
  // std::vector<std::vector<float>> boxes2;
  // for(int i=0;i<idx_th.size();i++){
  //   std::vector<float> temp;
  //   temp.push_back(boxes.at<float>(idx_th[i],0));
  //   temp.push_back(boxes.at<float>(idx_th[i],1));
  //   temp.push_back(boxes.at<float>(idx_th[i],2));
  //   temp.push_back(boxes.at<float>(idx_th[i],3));
  
  //   boxes2.push_back(temp);
  //   // temp.clear();
  // }

  // //good above
  // std::vector<float> scores3;
  // for(int i=0;i<idx_th.size();i++){
  //   scores3.push_back(scores2.at<float>(1,idx_th[i]));
  // }

  // std::vector<int> preds;
  // for(int i=0;i<idx_th.size();i++){
  //   preds.push_back(static_cast<int>(index.at<float>(1,idx_th[i])));
  // }
  
  //   // Print preds
  // std::cout << "preds: ";
  // for (const int& value : preds) {
  //     std::cout << value << " ";
  // }
  // std::cout << std::endl;

  // // Print scores3
  // std::cout << "scores3: ";
  // for (const float& value : scores3) {
  //     std::cout << value << " ";
  // }
  // std::cout << std::endl;

  // // Print boxes2
  // std::cout << "boxes2:" << std::endl;
  // for (const std::vector<float>& box : boxes2) {
  //     for (const float& value : box) {
  //         std::cout << value << " ";
  //     }
  //     std::cout << std::endl;
  // }
  
  
  //***approach1 and apraoch2 is the same results, but approach1 is way more faster
  




  // std::vector<std::vector<float>> boxes3=boxes2;

  // std::cout << boxes2.size() << std::endl;
  // boxes2[0]=xywh2xyxy_scale(boxes2[0],640.0, 640.0);
  // std::vector<float> scaled_box = xywh2xyxy_scale(boxes2[0], 640.0, 640.0);
  // boxes2[0] = scaled_box;
  // for(int q=0;q<boxes2.size();q++){
  //     boxes2[q]=xywh2xyxy_scale(boxes2[q],640.0, 640.0);
  //     // std::cout<< "=======" << std::endl;
  // }


  //so i have preds, scores3, boxes2, idx_th

  // std::vector<std::vector<int>> ind;
  // for(int i=0;i< 80;i++){
  //   std::vector<int> temp4;
  //   for(int j=0;j<preds.size();j++){
  //     if(i==preds[j]){
  //       std::cout <<preds[j] <<std::endl;
  //       temp4.push_back(j);
  //     }
  //   }

  //   if(temp4.empty()){
  //     temp4.push_back(81);
  //   }
  //   ind.push_back(temp4);
  //   // std::cout << temp4.size() << std::endl;

  // }

  // for(int i=0;i<ind.size();i++){
  //   if(ind[i][0]!=-1){
  //     for(int j=0;j<ind[i].size();j++){
  //       std::cout << ind[i][j] ;
  //     }
  //   }
  //   else{
  //     std::cout << ind[i][0];
  //   }
  //   std::cout << std::endl;
  // }
  // std::cout << ind.size() << std::endl;
  // // assert(ind.size()==80);
  // for(int i=0;i<80;i++){
  //   std::vector<std::vector<float>> selected_box;
  //   std::vector<int> indices;
  //   if(ind[i][0]!=81){
  //     for(int j=0;j<ind[i].size();j++){
  //       selected_box.push_back(boxes2[ind[i][j]]);
  //     }
  //     indices=NMS(selected_box,0.45);
  //     for(int a=0;a<indices.size();a++){
  //       std::cout << indices[a] <<std::endl;
  //     }
  //   }
  // }





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
cv::Scalar hex2rgb(const std::string& h) {
    return cv::Scalar(std::stoi(h.substr(1, 2), 0, 16), std::stoi(h.substr(3, 2), 0, 16), std::stoi(h.substr(5, 2), 0, 16));
}

cv::Scalar getColor(int i, bool bgr = false) {
    std::string hex[] = {
        "FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231", "48F90A", "92CC17", "3DDB86", "1A9334", "00D4BB",
        "2C99A8", "00C2FF", "344593", "6473FF", "0018EC", "8438FF", "520085", "CB38FF", "FF95C8", "FF37C7"
    };
    int n = sizeof(hex) / sizeof(hex[0]);
    cv::Scalar c = hex2rgb('#' + hex[i % n]);
    return bgr ? cv::Scalar(c[2], c[1], c[0]) : c;
}

cv::Mat plotOneBox(const std::vector<float>& x, cv::Mat im, cv::Scalar color = cv::Scalar(128, 128, 128),
                   const std::string& label = "", int lineThickness = 3) {
    int tl = lineThickness ? std::max(round(0.002 * (im.rows + im.cols) / 2), 1.0) : 1; // line/font thickness
    cv::Point c1(static_cast<int>(x[0]), static_cast<int>(x[1]));
    cv::Point c2(static_cast<int>(x[2]), static_cast<int>(x[3]));
    cv::rectangle(im, c1, c2, color, tl, cv::LINE_AA);
    if (!label.empty()) {
        int tf = std::max(tl - 1, 1); // font thickness
        cv::Size textSize = cv::getTextSize(label, 0, tl / 3, tf, nullptr);
        c2 = cv::Point(c1.x + textSize.width, c1.y - textSize.height - 3);
        cv::rectangle(im, c1, c2, color, -1, cv::LINE_AA); // filled
        cv::putText(im, label, cv::Point(c1.x, c1.y - 2), 0, tl / 3, cv::Scalar(225, 255, 255), tf, cv::LINE_AA);
    }
    return im;
}

void plotBboxes(const std::string& imgPath, const std::vector<std::vector<float>>& results,
                const std::vector<std::string>& coco_names, const std::string& savePath, 
                const std::map<int, std::vector<float>>& trackingData) {
    cv::Mat im0 = cv::imread(imgPath);
    
    for (int i = 0; i < results.size(); ++i) {
        const std::vector<float>& value = results[i];
        std::vector<float> bbox(value.begin(), value.begin() + 4);
        float confidence = value[4];
        int clsId = static_cast<int>(value[5]);
        std::string clsName = coco_names[clsId];

        // Retrieve the tracking ID from the trackingData map
        int trackingid=-1;
        
        auto it = trackingData.find(i);
        if(it != trackingData.end()){
          std::vector<float> foundValue = it->second;
          if (foundValue.size() > 0) {
              trackingid = static_cast<int>(foundValue[0]);
              // std::cout << foundValue[0] << std::endl;
          }
        }
        else{
          trackingid=-1;
        }
        // std::cout << trackingid << std::endl;
        // Include tracking ID, class name, and confidence in the label
        // std::string label = clsName + " " + std::to_string(trackingid) + " " + std::to_string(confidence);
        // std::string label = clsName + " "  + std::to_string(confidence);
        std::string label = clsName + " " + std::to_string(trackingid);
        cv::Scalar color = getColor(clsId, true);

        im0 = plotOneBox(bbox, im0, color, label);
    }

    try {
        cv::imwrite(savePath, im0);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

//above all works

//g++ -o yolov8_integer_tracking yolov8_integer_tracking.cpp `pkg-config --cflags --libs opencv4`
//./yolov8_integer_tracking

//simply copy objects here
std::vector<std::vector<float>> output_id(const std::string& img_path, const std::vector<std::vector<float>>& results){

    cv::Mat img = cv::imread(img_path);
    if (img.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        // You should return an empty cv::Mat or handle errors differently.
        return cv::Mat();
    }

    //must assume detected image and results are not 0 in width and height
    std::vector<std::vector<float>> unique_ids;


    int len_results=results.size();

    // cv::Mat unique_ids= cv::Mat::zeros(len_results, 27, CV_32F);
    //27 is ((10*(cls_id), b,g,r, confidence*100, x1/3,y1/3,x2/3,y2/3, b_detected/45, g_detected/45, r_detected/45))
    // 1,6,6,6,1,4,3,   

    for (int i =0; i< len_results; ++i){
        int cls_id=results[i][5];
        float confidence = results[i][4];

        vector <float> x;

        for (int j=0; j< 4; ++j){
            int each = static_cast<int>(std::round(results[i][j]));
            x.push_back(each);
        }

        cv::Mat detected=img(cv::Rect(x[0], x[1], x[2] - x[0], x[3] - x[1]));
        // cv::namedWindow("Detected Region", cv::WINDOW_NORMAL);
        // cv::imshow("Detected Region", detected);

        // // Wait for a key press and then close the window
        // cv::waitKey(0);
        // cv::destroyAllWindows();

        //detected.cols is height .shape[1]
        //detected.rows is width  .shape[0]
        // std::cout << detected.cols << std::endl;
        // std::cout << detected.rows << std::endl;

        if (detected.rows == 0 || detected.cols == 0) {
            // Report an error or handle the case where either rows or cols is 0
            std::cerr << "Error: detected.rows or detected.cols is 0." << std::endl;
            // Optionally, you can exit the program or take appropriate action here.
        }

        
        int split=2;
        float block_width=detected.rows/split;
        float block_height=detected.cols/split;

        // Create a vector to store the blocks
        std::vector<cv::Mat> blocks;

        // Split the image into blocks
        for (int i = 0; i < split; ++i) {
            for (int j = 0; j < split; ++j) {
                int x1 = j * block_width;
                int y1 = i * block_height;
                int x2 = (j + 1) * block_width;
                int y2 = (i + 1) * block_height;

                // Ensure the coordinates are within bounds
                x2 = std::min(x2, detected.rows);
                y2 = std::min(y2, detected.cols);

                // Extract the block
                cv::Mat block = detected(cv::Rect(y1, x1, y2 - y1, x2 - x1));
                // cv::imshow("Detected Region", block);

                // // Wait for a key press and then close the window
                // cv::waitKey(0);
                // cv::destroyAllWindows();

                blocks.push_back(block);
            }
        }

        // Separate B, G, and R channels of each block


        std::vector<cv::Mat> b, g, r;
        for (const cv::Mat& block : blocks) {
            std::vector<cv::Mat> channels(3);
            cv::split(block, channels);
            
            b.push_back(channels[0]);
            g.push_back(channels[1]);
            r.push_back(channels[2]);
        }

        // Calculate variance and sort for each channel
        std::vector<int> b_var, g_var, r_var;

        for (size_t i = 0; i < b.size(); ++i) {
            
            for (size_t j = i + 1; j < b.size(); ++j) {

                cv::Scalar bi_mean, gi_mean, ri_mean;
                cv::Scalar bi_stddev, gi_stddev, ri_stddev;
                cv::Scalar bj_mean, gj_mean, rj_mean;
                cv::Scalar bj_stddev, gj_stddev, rj_stddev;


                cv::meanStdDev(b[i], bi_mean, bi_stddev);
                cv::meanStdDev(b[j], bj_mean, bj_stddev);
                cv::meanStdDev(g[i], gi_mean, gi_stddev);
                cv::meanStdDev(g[j], gj_mean, gj_stddev);
                cv::meanStdDev(r[i], ri_mean, ri_stddev);
                cv::meanStdDev(r[j], rj_mean, rj_stddev);


                double bi_var = bi_stddev[0] * bi_stddev[0];
                double bj_var = bj_stddev[0] * bj_stddev[0];
                double b_var_result=bi_var*bj_var;

                double gi_var = gi_stddev[0] * gi_stddev[0];
                double gj_var = gj_stddev[0] * gj_stddev[0];
                double g_var_result=gi_var*gj_var;

                double ri_var = ri_stddev[0] * ri_stddev[0];
                double rj_var = rj_stddev[0] * rj_stddev[0];
                double r_var_result=ri_var*rj_var;


                b_var.push_back(static_cast<int>(b_var_result / (detected.rows * detected.cols)));
                g_var.push_back(static_cast<int>(g_var_result / (detected.rows * detected.cols)));
                r_var.push_back(static_cast<int>(r_var_result / (detected.rows * detected.cols)));


            }
        }

        // Sort the variance vectors
        std::sort(b_var.begin(), b_var.end());
        std::sort(g_var.begin(), g_var.end());
        std::sort(r_var.begin(), r_var.end());


        // // Printing b_var
        // std::cout << "b_var: ";
        // for (int i = 0; i < b_var.size(); ++i) {
        //     std::cout << b_var[i] << " ";
        // }
        // std::cout << std::endl;

        // // Printing g_var
        // std::cout << "g_var: ";
        // for (int i = 0; i < g_var.size(); ++i) {
        //     std::cout << g_var[i] << " ";
        // }
        // std::cout << std::endl;

        // // Printing r_var
        // std::cout << "r_var: ";
        // for (int i = 0; i < r_var.size(); ++i) {
        //     std::cout << r_var[i] << " ";
        // }
        // std::cout << std::endl;


        //so b_var, g_var, r_var is std::vector<int> b_var, g_var, r_var;

        // Calculate the maximum and minimum values for b, g, and r
        double b_max = *std::max_element(b_var.begin(), b_var.end());
        double b_min = *std::min_element(b_var.begin(), b_var.end());

        double g_max = *std::max_element(g_var.begin(), g_var.end());
        double g_min = *std::min_element(g_var.begin(), g_var.end());

        double r_max = *std::max_element(r_var.begin(), r_var.end());
        double r_min = *std::min_element(r_var.begin(), r_var.end());

        // Calculate intervals
        double b_interval = (b_max - b_min) / (b_var.size() - 1);
        double g_interval = (g_max - g_min) / (g_var.size() - 1);
        double r_interval = (r_max - r_min) / (r_var.size() - 1);

        // Normalize the values in b_var, g_var, and r_var
        for (size_t i = 0; i < b_var.size(); ++i) {
            if (b_interval != 0) {
                b_var[i] = (b_var[i] - b_min) / b_interval + 1;
            } else {
                // Handle the case where b_interval is zero (or any other invalid value)
                // You can assign a default value or raise an exception depending on your logic.
                b_var[i] = 0;
            }
        }

        for (size_t i = 0; i < g_var.size(); ++i) {
            if (g_interval != 0) {
                g_var[i] = (g_var[i] - g_min) / g_interval + 1;
            } else {
                g_var[i] = 0;
            }
        }

        for (size_t i = 0; i < r_var.size(); ++i) {
            if (r_interval != 0) {
                r_var[i] = (r_var[i] - r_min) / r_interval + 1;
            } else {
                r_var[i] = 0;
            }
        }


        cv::Mat b_detected;
        cv::Mat g_detected;
        cv::Mat r_detected;    

        std::vector<cv::Mat> channels_detected(3);
        cv::split(detected, channels_detected);
        
        b_detected=channels_detected[0];
        g_detected=channels_detected[1];
        r_detected=channels_detected[2];


        cv::Scalar b_mean, g_mean, r_mean;
        cv::Scalar b_stddev, g_stddev, r_stddev;

        cv::meanStdDev(b_detected, b_mean, b_stddev);
        cv::meanStdDev(g_detected, g_mean, g_stddev);
        cv::meanStdDev(r_detected, r_mean, r_stddev);
        //just use b_mean/45, g_mean/45, r_mean/45

        std::vector<float> combinedData;
        combinedData.push_back(static_cast<float>(cls_id*10));

        for (const int& value : b_var) {
            combinedData.push_back(static_cast<float>(value));
        }
        for (const int& value : g_var) {
            combinedData.push_back(static_cast<float>(value));
        }
        for (const int& value : r_var) {
            combinedData.push_back(static_cast<float>(value));
        }

        combinedData.push_back(confidence*100);

        combinedData.push_back(x[0]/3);
        combinedData.push_back(x[1]/3);
        combinedData.push_back(x[2]/3);
        combinedData.push_back(x[3]/3);

        // std::cout << b_detected.rows << std::endl;
        // std::cout << b_detected.cols << std::endl;

        combinedData.push_back(static_cast<float>(b_stddev[0]*b_stddev[0]/45));
        combinedData.push_back(static_cast<float>(g_stddev[0]*g_stddev[0]/45));
        combinedData.push_back(static_cast<float>(r_stddev[0]*r_stddev[0]/45));

        // std::cout << b_mean[0] << std::endl;
        // std::cout << g_mean[0] << std::endl;
        // std::cout << r_mean[0] << std::endl;


        // std::cout << "combinedData: ";
        // for (size_t i = 0; i < combinedData.size(); ++i) {
        //     std::cout << combinedData[i] << " ";
        // }
        // std::cout << std::endl;
        unique_ids.push_back(combinedData);

    }//end of for loop



    // std::cout << "unique_id: ";
    // for (size_t i = 0; i < unique_ids.size(); ++i) {
    //     std::cout << unique_ids[i] << " ";
    // }
    // std::cout << std::endl;

    return unique_ids;

}




//start the second function:

//not correct valid
cv::Mat normalize(const cv::Mat& image) {
    cv::Mat im(image.rows, image.cols, CV_32FC3);

    for (int x = 0; x < image.rows; ++x) {
        for (int y = 0; y < image.cols; ++y) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(x, y);
            float div = std::max(std::max(pixel[0], pixel[1]), pixel[2]);

            if (div == 0) {
                div = 1;
            }

            cv::Vec3f normalized_pixel;
            normalized_pixel[0] = static_cast<float>(pixel[0]) / div;
            normalized_pixel[1] = static_cast<float>(pixel[1]) / div;
            normalized_pixel[2] = static_cast<float>(pixel[2]) / div;

            im.at<cv::Vec3f>(x, y) = normalized_pixel;
        }
    }

    return im;
}

// Define a function to calculate SVD (you should implement this function)
std::vector<cv::Mat> calculateSVD(const cv::Mat& detected) {
    cv::Mat l1, l2, l3;
    cv::Mat l4;

    cv::Mat image1 = normalize(detected);
    int height = image1.rows;
    int width = image1.cols;

    cv::Mat rc = image1.clone();
    cv::Mat gc = image1.clone();
    cv::Mat bc = image1.clone();

    // Extract color channels
    cv::Mat channels[3];
    cv::split(image1, channels);
    rc = channels[0];
    gc = channels[1];
    bc = channels[2];
    

    // Perform SVD on each channel
    cv::SVD svd_rc(rc, cv::SVD::FULL_UV);
    cv::SVD svd_gc(gc, cv::SVD::FULL_UV);
    cv::SVD svd_bc(bc, cv::SVD::FULL_UV);
    // cv::SVD svd_rc(rc, cv::SVD::MODIFY_A);
    // cv::SVD svd_gc(gc, cv::SVD::MODIFY_A);
    // cv::SVD svd_bc(bc, cv::SVD::MODIFY_A);
    // Get the first 5 singular values for each channel
    cv::Mat s_rc, s_gc, s_bc;


    if (svd_rc.w.rows >= 5) {
        s_rc = svd_rc.w.rowRange(0, 5);
        l1=s_rc;
    } else {
        // Handle the case when the matrix doesn't have enough rows
        // You may want to add error handling or return empty matrices here
    }

    if (svd_gc.w.rows >= 5) {
        s_gc = svd_gc.w.rowRange(0, 5);
        l2=s_gc;
    } else {
        // Handle the case when the matrix doesn't have enough rows
        // You may want to add error handling or return empty matrices here
    }

    if (svd_bc.w.rows >= 5) {
        s_bc = svd_bc.w.rowRange(0, 5);
        l3=s_bc;
    } else {
        // Handle the case when the matrix doesn't have enough rows
        // You may want to add error handling or return empty matrices here
    }

    // Store image dimensions in l4
    l4= cv::Mat(1, 2, CV_32S);
    l4.at<int>(0, 0) = height;
    l4.at<int>(0, 1) = width;

    std::vector<cv::Mat> result;

    result.push_back(l1);
    result.push_back(l2);
    result.push_back(l3);
    result.push_back(l4);



    return result;

}


double get_score(
    const cv::Mat& l1, const cv::Mat& l2, const cv::Mat& l3, const cv::Mat& l4,
    const cv::Mat& c1, const cv::Mat& c2, const cv::Mat& c3, const cv::Mat& c4) {
    
    float sum = 0;
    float mag2 = 0;
    // int mag2_int=0;

    // std::cout << l1.rows << std::endl;
    // std::cout << l1.cols << std::endl;

    for (int x = 0; x < l1.rows; ++x) {
        // std::cout << l1.at<float>(x, 0) << std::endl;
        sum += pow(pow(l1.at<float>(x, 0) - c1.at<float>(x, 0), 2) +
                    pow(l2.at<float>(x,0) - c2.at<float>(x,0), 2) +
                    pow(l3.at<float>(x,0) - c3.at<float>(x,0), 2),0.5);
        // std::cout << sum << std::endl;
    }

    // std::cout << l4.rows << std::endl;
    // std::cout << l4.cols << std::endl;
    // std::cout << l4 << std::endl;
    // std::cout << l4.at<int>(0, 0) << std::endl;
    // std::cout << l4.at<int>(0, 1) << std::endl;
    for (int x = 0; x < l4.rows; ++x) {
        // std::cout << static_cast<float>(l4.at<int>(0, x)) << std::endl;

        mag2 += (pow(static_cast<float>(l4.at<int>(0, x)) - static_cast<float>(c4.at<int>(0, x)), 2));
        // mag2_int+= pow(l4.at<int>(0, x) - c4.at<int>(0, x), 2);
    }


    std::cout << mag2<< std::endl;

    mag2 = mag2 / pow((static_cast<float>(l4.at<int>(0, 0))  * static_cast<float>(l4.at<int>(0, 1)) )
     + (static_cast<float>(c4.at<int>(0, 0)) * static_cast<float>(c4.at<int>(0, 1))), 0.5);
    std::cout << mag2 << std::endl;
    float mag1 = pow(sum, 0.5);
    mag2 = pow(mag2, 0.5);

    // std::cout << mag1 << std::endl;
    // std::cout << mag2 << std::endl;
    return mag1 + mag2;
    // return 1;
}



std::map<int, std::vector<float>> generateIds(const std::vector<std::vector<float>>& unique_ids) {
    std::map<int, std::vector<float>> ids;
    for (int i = 0; i < unique_ids.size(); ++i) {
        ids[i] = {static_cast<float>(i), -1.0f, static_cast<float>(unique_ids[i][0] / 10.0)};
    }
    return ids;
}

std::vector<std::map<int, std::vector<float>>> compare(
    const std::string& img_path1, const std::vector<std::vector<float>>& results1,
    const std::vector<std::vector<float>>& unique_ids1,
    const std::string& img_path2, const std::vector<std::vector<float>>& results2,
    const std::vector<std::vector<float>>& unique_ids2) 
{

    std::vector<std::map<int, std::vector<float>>> result;

    cv::Mat image1 = cv::imread(img_path1);
    if (image1.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        // You should return an empty cv::Mat or handle errors differently.
        return std::vector<std::map<int, std::vector<float>>>();
    }

    cv::Mat image2 = cv::imread(img_path2);
    if (image2.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        // You should return an empty cv::Mat or handle errors differently.
        return std::vector<std::map<int, std::vector<float>>>();
    }

    int svd_threshold = 8;
    int cut_threshold = 40;

    std::map<int, std::vector<float>> ids1;
    std::map<int, std::vector<float>> ids2;
    
    if (unique_ids1.size() > unique_ids2.size()) {
        // std::cout << "-------------------------" << std::endl;
        ids1 = generateIds(unique_ids1);
        // std::cout << "==============================" << std::endl;

        int addition = 1;

        // // Iterate through the map and print the key-value pairs
        // for (const auto& entry : ids2) {
        //     int key = entry.first;
        //     const std::vector<float>& values = entry.second;

        //     std::cout << "Key: " << key << ", Values: ";
            
        //     for (float value : values) {
        //         std::cout << value << " ";
        //     }

        //     std::cout << std::endl;
        // }        

        // std::cout << "-------------------------" << std::endl;
        // Iterate through the vectors in unique_ids1
        for (size_t i = 0; i < unique_ids2.size(); ++i) {
            float min_norm2 = std::numeric_limits<float>::infinity();
            int matching_id2 = -1;

            // Compare with vectors in unique_ids2
            for (size_t j = 0; j < unique_ids1.size(); ++j) {
                if (unique_ids2[i][0] == unique_ids1[j][0]) {
                    float norm2 = 0.0;
                    for (size_t k = 1; k < unique_ids2[i].size(); ++k) {
                        norm2 += std::pow(unique_ids1[j][k] - unique_ids2[i][k], 2);
                    }
                    norm2 = std::sqrt(norm2);

                    if (norm2 < min_norm2) {
                        min_norm2 = norm2;
                        matching_id2 = static_cast<int>(j);
                    }
                }
            }

            if (cut_threshold > min_norm2) {
                ids2[i] = {static_cast<float>(matching_id2), min_norm2, unique_ids2[i][0] / 10};
                ids1[matching_id2][1] = 1.0;
            } else {
                ids2[i] = {-1, -1, unique_ids1[i][0] / 10};
            }
        }

        // std::cout << "==============================" << std::endl;
        // ids1[0][0]=3;



        // // Iterate through the map and print the key-value pairs
        // for (const auto& entry : ids1) {
        //     int key = entry.first;
        //     const std::vector<float>& values = entry.second;

        //     std::cout << "Key: " << key << ", Values: ";
            
        //     for (float value : values) {
        //         std::cout << value << " ";
        //     }

        //     std::cout << std::endl;
        // }    

        // // Iterate through the map and print the key-value pairs
        // for (const auto& entry : ids2) {
        //     int key = entry.first;
        //     const std::vector<float>& values = entry.second;

        //     std::cout << "Key: " << key << ", Values: ";
            
        //     for (float value : values) {
        //         std::cout << value << " ";
        //     }

        //     std::cout << std::endl;
        // }    
        // std::cout << "-------------------------" << std::endl;
        
        


        for (auto it1 = ids2.begin(); it1 != ids2.end(); ++it1) {
            for (auto it2 = ids2.begin(); it2 != ids2.end(); ++it2) {
                int key1 = it1->first;
                int key2 = it2->first;
                std::vector<float>& value1 = it1->second;
                std::vector<float>& value2 = it2->second;

                if (key1 != key2 && value1[0] == value2[0]) {
                    if (value1[2] == value2[2]) {
                        if (value1[1] != -1 && value2[1] != -1) {
                            if (value1[1] > value2[1]) {
                                value2[0] = ids2.size() + addition;
                                addition++;
                            } else {
                                value1[0] = ids2.size() + addition;
                                addition++;
                            }
                        }
                    } else {
                        if (value1[1] != -1 && value2[1] != -1) {
                            value1[0] = ids2.size() + addition + 1;
                            addition++;
                        }
                    }
                }
            }
        }


        // // Iterate through the map and print the key-value pairs
        // for (const auto& entry : ids1) {
        //     int key = entry.first;
        //     const std::vector<float>& values = entry.second;

        //     std::cout << "Key: " << key << ", Values: ";
            
        //     for (float value : values) {
        //         std::cout << value << " ";
        //     }

        //     std::cout << std::endl;
        // }    

        // // Iterate through the map and print the key-value pairs
        // for (const auto& entry : ids2) {
        //     int key = entry.first;
        //     const std::vector<float>& values = entry.second;

        //     std::cout << "Key: " << key << ", Values: ";
            
        //     for (float value : values) {
        //         std::cout << value << " ";
        //     }

        //     std::cout << std::endl;
        // }    

        for (size_t i = 0; i < ids2.size(); ++i) {
            if (ids2[i][0] == -1) {

                vector <float> x;

                for (int g=0; g< 4; ++g){
                    int each2 = static_cast<int>(std::round(results2[i][g]));
                    x.push_back(each2);
                }

                cv::Mat detected2 = image2(cv::Rect(x[0], x[1], x[2] - x[0], x[3] - x[1]));

                float class_id2 = results2[i][5];

                int index = -1;
                float min_score = 200.0;
        
                for (size_t j = 0; j < ids1.size(); ++j) {
                    if (ids1[j][1] == -1) {
      
                        vector <float> x2;
                        for (int k=0; k< 4; ++k){
                            int each = static_cast<int>(std::round(results1[j][k]));
                            x2.push_back(each);
                        }

                        cv::Mat detected1 = image1(cv::Rect(x2[0], x2[1], x2[2] - x2[0], x2[3] - x2[1]));

                        float class_id1 = results1[j][5];
      

                        cv::Mat ds_detected2;
                        cv::resize(detected2, ds_detected2, cv::Size(30, 30));

                        std::vector<cv::Mat> l1l4 = calculateSVD(ds_detected2);

                    //    for (size_t i = 0; i < l1l4.size(); ++i) {
                    //         std::cout << "Matrix " << i << ":" << std::endl;
                    //         std::cout << l1l4[i] << std::endl;
                    //     }


                        

                        if (class_id2 == class_id1){
                            cv::Mat ds_detected1;
                            cv::resize(detected1, ds_detected1, cv::Size(30, 30));
                            std::vector<cv::Mat> c1c4 = calculateSVD(ds_detected1);

                                //it is correct now for both l1l4 and c1c4

                        //     for (size_t i = 0; i < c1c4.size(); ++i) {
                        //             std::cout << "Matrix " << i << ":" << std::endl;
                        //             std::cout << c1c4[i] << std::endl;
                        //         }
                            
                            cv::Mat l1= l1l4[0];
                            cv::Mat l2= l1l4[1];
                            cv::Mat l3= l1l4[2];
                            cv::Mat l4= l1l4[3];

                            cv::Mat c1=c1c4[0];
                            cv::Mat c2=c1c4[1];
                            cv::Mat c3=c1c4[2];
                            cv::Mat c4=c1c4[3];

                            // std::cout << l1l4[0].size() << std::endl;

                            float ms = get_score(l1,l2,l3,l4,c1,c2,c3,c4);
                            // std::cout << ms<< std::endl;
                            if (ms < min_score) {
                                min_score = ms;
                                index = j;
                            }


                            if (min_score < svd_threshold) {
                   
                                ids2[i][0] = index;
                                ids2[i][1] = -2;
                                ids2[i].push_back(min_score);
                            }




                        }



                    }
                }
            }
        }

        //now delete duplicate ones
        for(int q=0;q< ids2.size(); q++){
          if(ids2[q][1]==-2){
            int min_index=q;
            int goal=ids2[q][0];
            for(int p=q+1;p<ids2.size(); p++){
              if(ids2[p][1]==-2 && ids2[p][0]==ids2[q][0]){
                if(ids2[q][3]<=ids2[p][3]){
                  ids2[p][0]=ids2.size()+addition+1;
                  addition++;
                  min_index=q;
                }else{
                  ids2[q][0]=ids2.size()+addition+1;
                  addition++;
                  min_index=p;
                }
              }
            }
            ids2[min_index][0]=goal;
          }
        }

     
        // std::cout << "==============================" << std::endl;



        // // Iterate through the map and print the key-value pairs
        // for (const auto& entry : ids1) {
        //     int key = entry.first;
        //     const std::vector<float>& values = entry.second;

        //     std::cout << "Key: " << key << ", Values: ";
            
        //     for (float value : values) {
        //         std::cout << value << " ";
        //     }

        //     std::cout << std::endl;
        // }    

        // // Iterate through the map and print the key-value pairs
        // for (const auto& entry : ids2) {
        //     int key = entry.first;
        //     const std::vector<float>& values = entry.second;

        //     std::cout << "Key: " << key << ", Values: ";
            
        //     for (float value : values) {
        //         std::cout << value << " ";
        //     }

        //     std::cout << std::endl;
        // }    

    }
    else {
        

        ids2 = generateIds(unique_ids2);

        int addition = 1;

        // // Iterate through the map and print the key-value pairs
        // for (const auto& entry : ids2) {
        //     int key = entry.first;
        //     const std::vector<float>& values = entry.second;

        //     std::cout << "Key: " << key << ", Values: ";
            
        //     for (float value : values) {
        //         std::cout << value << " ";
        //     }

        //     std::cout << std::endl;
        // }        

        
        // Iterate through the vectors in unique_ids1
        for (size_t i = 0; i < unique_ids1.size(); ++i) {
            float min_norm1 = std::numeric_limits<float>::infinity();
            int matching_id = -1;

            // Compare with vectors in unique_ids2
            for (size_t j = 0; j < unique_ids2.size(); ++j) {
                if (unique_ids1[i][0] == unique_ids2[j][0]) {
                    float norm = 0.0;
                    for (size_t k = 1; k < unique_ids1[i].size(); ++k) {
                        norm += std::pow(unique_ids2[j][k] - unique_ids1[i][k], 2);
                    }
                    norm = std::sqrt(norm);

                    if (norm < min_norm1) {
                        min_norm1 = norm;
                        matching_id = static_cast<int>(j);
                    }
                }
            }

            if (cut_threshold > min_norm1) {
                ids1[i] = {static_cast<float>(matching_id), min_norm1, unique_ids1[i][0] / 10};
                ids2[matching_id][1] = 1.0;
            } else {
                ids1[i] = {-1, -1, unique_ids1[i][0] / 10};
            }
        }


        // ids1[0][0]=3;



        // // Iterate through the map and print the key-value pairs
        // for (const auto& entry : ids1) {
        //     int key = entry.first;
        //     const std::vector<float>& values = entry.second;

        //     std::cout << "Key: " << key << ", Values: ";
            
        //     for (float value : values) {
        //         std::cout << value << " ";
        //     }

        //     std::cout << std::endl;
        // }    

        // // Iterate through the map and print the key-value pairs
        // for (const auto& entry : ids2) {
        //     int key = entry.first;
        //     const std::vector<float>& values = entry.second;

        //     std::cout << "Key: " << key << ", Values: ";
            
        //     for (float value : values) {
        //         std::cout << value << " ";
        //     }

        //     std::cout << std::endl;
        // }    

        
        


        for (auto it1 = ids1.begin(); it1 != ids1.end(); ++it1) {
            for (auto it2 = ids1.begin(); it2 != ids1.end(); ++it2) {
                int key1 = it1->first;
                int key2 = it2->first;
                std::vector<float>& value1 = it1->second;
                std::vector<float>& value2 = it2->second;

                if (key1 != key2 && value1[0] == value2[0]) {
                    if (value1[2] == value2[2]) {
                        if (value1[1] != -1 && value2[1] != -1) {
                            if (value1[1] > value2[1]) {
                                value2[0] = ids1.size() + addition;
                                addition++;
                            } else {
                                value1[0] = ids1.size() + addition;
                                addition++;
                            }
                        }
                    } else {
                        if (value1[1] != -1 && value2[1] != -1) {
                            value1[0] = ids1.size() + addition + 1;
                            addition++;
                        }
                    }
                }
            }
        }

        

        // // Iterate through the map and print the key-value pairs
        // for (const auto& entry : ids1) {
        //     int key = entry.first;
        //     const std::vector<float>& values = entry.second;

        //     std::cout << "Key: " << key << ", Values: ";
            
        //     for (float value : values) {
        //         std::cout << value << " ";
        //     }

        //     std::cout << std::endl;
        // }    

        // // Iterate through the map and print the key-value pairs
        // for (const auto& entry : ids2) {
        //     int key = entry.first;
        //     const std::vector<float>& values = entry.second;

        //     std::cout << "Key: " << key << ", Values: ";
            
        //     for (float value : values) {
        //         std::cout << value << " ";
        //     }

        //     std::cout << std::endl;
        // }    

        for (size_t i = 0; i < ids1.size(); ++i) {
            if (ids1[i][0] == -1) {

                vector <float> x;

                for (int g=0; g< 4; ++g){
                    int each = static_cast<int>(std::round(results1[i][g]));
                    x.push_back(each);
                }

                cv::Mat detected1 = image1(cv::Rect(x[0], x[1], x[2] - x[0], x[3] - x[1]));

                float class_id1 = results1[i][5];

                int index = -1;
                float min_score = 200.0;

                for (size_t j = 0; j < ids2.size(); ++j) {
                    if (ids2[j][1] == -1) {
                        // std::cout << "------------" << std::endl;
                        vector <float> x2;
                        for (int k=0; k< 4; ++k){
                            int each2 = static_cast<int>(std::round(results2[j][k]));
                            x2.push_back(each2);
                        }

                        cv::Mat detected2 = image2(cv::Rect(x2[0], x2[1], x2[2] - x2[0], x2[3] - x2[1]));

                        float class_id2 = results2[j][5];
                        cv::Mat ds_detected1;
                        cv::resize(detected1, ds_detected1, cv::Size(30, 30));
                        std::vector<cv::Mat> l1l4 = calculateSVD(ds_detected1);

                    //    for (size_t i = 0; i < l1l4.size(); ++i) {
                    //         std::cout << "Matrix " << i << ":" << std::endl;
                    //         std::cout << l1l4[i] << std::endl;
                    //     }

                        // std::cout << "------------" << std::endl;
                        

                        if (class_id1 == class_id2){
                            cv::Mat ds_detected2;
                            cv::resize(detected2, ds_detected2, cv::Size(30, 30));
                            std::vector<cv::Mat> c1c4 = calculateSVD(ds_detected2);

                                //it is correct now for both l1l4 and c1c4

                        //     for (size_t i = 0; i < c1c4.size(); ++i) {
                        //             std::cout << "Matrix " << i << ":" << std::endl;
                        //             std::cout << c1c4[i] << std::endl;
                        //         }
                            
                            cv::Mat l1= l1l4[0];
                            cv::Mat l2= l1l4[1];
                            cv::Mat l3= l1l4[2];
                            cv::Mat l4= l1l4[3];

                            cv::Mat c1=c1c4[0];
                            cv::Mat c2=c1c4[1];
                            cv::Mat c3=c1c4[2];
                            cv::Mat c4=c1c4[3];

                            // std::cout << l1l4[0].size() << std::endl;

                            float ms = get_score(l1,l2,l3,l4,c1,c2,c3,c4);
                            // std::cout << ms<< std::endl;
                            if (ms < min_score) {
                                min_score = ms;
                                index = j;
                            }


                            if (min_score < svd_threshold) {
                                // std::cout << "-------------------------" << std::endl;
                                ids1[i][0] = index;
                                ids1[i][1] = -2;
                                ids1[i].push_back(min_score);
                            }




                        }



                    }
                }
            }
        }

        //delete duplicate after svd
        for(int q=0;q< ids1.size(); q++){
          if(ids1[q][1]==-2){
            int min_index=q;
            int goal=ids1[q][0];
            for(int p=q+1;p<ids1.size(); p++){
              if(ids1[p][1]==-2 && ids1[p][0]==ids1[q][0]){
                if(ids1[q][3]<=ids1[p][3]){
                  ids1[p][0]=ids1.size()+addition+1;
                  addition++;
                  min_index=q;
                }else{
                  ids1[q][0]=ids1.size()+addition+1;
                  addition++;
                  min_index=p;
                }
              }
            }
            ids1[min_index][0]=goal;
          }
        }






 


    }


    // // Iterate through the map and print the key-value pairs
    // for (const auto& entry : ids1) {
    //     int key = entry.first;
    //     const std::vector<float>& values = entry.second;

    //     std::cout << "Key: " << key << ", Values: ";
        
    //     for (float value : values) {
    //         std::cout << value << " ";
    //     }

    //     std::cout << std::endl;
    // }    

    // // Iterate through the map and print the key-value pairs
    // for (const auto& entry : ids2) {
    //     int key = entry.first;
    //     const std::vector<float>& values = entry.second;

    //     std::cout << "Key: " << key << ", Values: ";
        
    //     for (float value : values) {
    //         std::cout << value << " ";
    //     }

    //     std::cout << std::endl;
    // }    

    result.push_back(ids1);
    result.push_back(ids2);

    return result;

}


std::vector<int> motion_detection_pair(const std::vector<vector<float>>& results1, const std::map<int, std::vector<float>>& ids1,
const std::vector<vector<float>>& results2, const std::map<int, std::vector<float>>& ids2, int move_threshold, float ratio_threshold){

  std::vector<int> results;

  assert(results1.size()==ids1.size());
  assert(results2.size()==ids2.size());

  if(results1.size()>results2.size()){
    //ids1 is the longer one with [0] euqal to index
    //then traverse ids2
    // Traverse the map using a range-based for loop
    for(int i=0;i< results2.size(); i++){
      auto it= ids2.find(i);
      if(it!=ids2.end()){
        const std::vector<float>& values = it->second;//i is the index at ids2
        int corresp=static_cast<int>(values[0]);//corresp is the index at ids1

        //now the index is i and corresp, compare the size
        bool moved=false;
        if(corresp!=-1){
          std::vector<float> i_box;//ids2
          std::vector<float> corresp_box;//ids1

          for(int m=0;m<4;m++){
            int each2=static_cast<int>(std::round(results2[i][m]));
            i_box.push_back(each2);
          }

          for(int m=0;m<4;m++){
            int each2=static_cast<int>(std::round(results1[corresp][m]));
            corresp_box.push_back(each2);
          }

          //check moved
          int i_x_mid=static_cast<int>(std::round(i_box[2]-i_box[0])/2)+i_box[0];
          int i_y_mid=static_cast<int>(std::round(i_box[3]-i_box[1])/2)+i_box[1];

          int corresp_x_mid=static_cast<int>(std::round(corresp_box[2]-corresp_box[0])/2)+corresp_box[0];
          int corresp_y_mid=static_cast<int>(std::round(corresp_box[3]-corresp_box[1])/2)+corresp_box[1];

          
          float i_ratio=(i_box[2]-i_box[0])/(i_box[3]-i_box[1]);

          float corresp_ratio=(corresp_box[2]-corresp_box[0])/(corresp_box[3]-corresp_box[1]);

          if((i_ratio/corresp_ratio>ratio_threshold)&&((std::abs(corresp_x_mid-i_x_mid)>move_threshold)||(std::abs(corresp_y_mid-i_y_mid)>move_threshold))){
            moved=true;
          }
          // std::cout << std::abs((i_box[2]-i_box[0])*(i_box[3]-i_box[1])) << std::abs((corresp_box[2]-corresp_box[0])*(corresp_box[3]-corresp_box[1])) << std::endl;
          // assert(1==0);
          if((0.7> std::abs(std::abs((i_box[2]-i_box[0])*(i_box[3]-i_box[1]))/std::abs((corresp_box[2]-corresp_box[0])*(corresp_box[3]-corresp_box[1]))))||
          (1.3< std::abs(std::abs((i_box[2]-i_box[0])*(i_box[3]-i_box[1]))/std::abs((corresp_box[2]-corresp_box[0])*(corresp_box[3]-corresp_box[1]))))){
            moved=false;
          }
          std::cout << "class is: "<< " " << corresp << " " << std::abs(corresp_x_mid-i_x_mid) << " " << std::abs(corresp_y_mid-i_y_mid) << " " << i_ratio/corresp_ratio << " "<< std::abs(std::abs((i_box[2]-i_box[0])*(i_box[3]-i_box[1]))/std::abs((corresp_box[2]-corresp_box[0])*(corresp_box[3]-corresp_box[1]))) << std::endl;
        }

        if(moved){
          results.push_back(corresp);//i is the same with corresp
        }
      }
    }

  }else{
    //ids2 is [i]==index
    //travers ids1

    for(int i=0;i< results1.size(); i++){
      auto it= ids1.find(i);
      if(it!=ids1.end()){
        const std::vector<float>& values = it->second;//i is the index at ids1
        int corresp=static_cast<int>(values[0]);//corresp is the index at ids2

        //now the index is i and corresp, compare the size
        bool moved=false;
        if(corresp!=-1){
          std::vector<float> i_box;//ids1
          std::vector<float> corresp_box;//ids2

          for(int m=0;m<4;m++){
            int each2=static_cast<int>(std::round(results1[i][m]));
            i_box.push_back(each2);
          }

          for(int m=0;m<4;m++){
            int each2=static_cast<int>(std::round(results2[corresp][m]));
            corresp_box.push_back(each2);
          }

          //check moved
          int i_x_mid=static_cast<int>(std::round(i_box[2]-i_box[0])/2)+i_box[0];
          int i_y_mid=static_cast<int>(std::round(i_box[3]-i_box[1])/2)+i_box[1];

          int corresp_x_mid=static_cast<int>(std::round(corresp_box[2]-corresp_box[0])/2)+corresp_box[0];
          int corresp_y_mid=static_cast<int>(std::round(corresp_box[3]-corresp_box[1])/2)+corresp_box[1];


          float i_ratio=(i_box[2]-i_box[0])/(i_box[3]-i_box[1]);

          float corresp_ratio=(corresp_box[2]-corresp_box[0])/(corresp_box[3]-corresp_box[1]);

          if((i_ratio/corresp_ratio>ratio_threshold)&& ((std::abs(corresp_x_mid-i_x_mid)>move_threshold)||(std::abs(corresp_y_mid-i_y_mid)>move_threshold))){
            moved=true;
          }
          if((0.7> std::abs(std::abs((i_box[2]-i_box[0])*(i_box[3]-i_box[1]))/std::abs((corresp_box[2]-corresp_box[0])*(corresp_box[3]-corresp_box[1]))))||
          (1.3< std::abs(std::abs((i_box[2]-i_box[0])*(i_box[3]-i_box[1]))/std::abs((corresp_box[2]-corresp_box[0])*(corresp_box[3]-corresp_box[1]))))){
            moved=false;
          }
          std::cout << "class is: "<< " " << corresp << " " << std::abs(corresp_x_mid-i_x_mid) << " " << std::abs(corresp_y_mid-i_y_mid) << " " << i_ratio/corresp_ratio << " "<< std::abs(std::abs((i_box[2]-i_box[0])*(i_box[3]-i_box[1]))/std::abs((corresp_box[2]-corresp_box[0])*(corresp_box[3]-corresp_box[1]))) << std::endl;
        }

        if(moved){
          results.push_back(corresp);//i is the same with corresp
        }


      }
    }
  }
  return results;
 
}




int main(int argc, char **argv)
{

std::vector<std::string> coco_names = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
};


    // create model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile("yolov8l_integer_quant.tflite");
  
  auto ext_delegate_option = TfLiteExternalDelegateOptionsDefault("/usr/lib/libvx_delegate.so");
  auto ext_delegate_ptr = TfLiteExternalDelegateCreate(&ext_delegate_option);
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  interpreter->ModifyGraphWithDelegate(ext_delegate_ptr);
  interpreter->SetAllowFp16PrecisionForFp32(false);
  interpreter->AllocateTensors();

  std::cout << " Tensorflow Test " << endl;

  
  // string imgf1 ="image0frame937.jpg";
  // string imgf2 ="image0frame900.jpg";

  string imgf1 ="./0030.jpg";
  string imgf2 ="./0060.jpg";

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
  cv::Mat img1_mat = convert_image(img1, 1080, 1920, filesize);
  char *img2 = read_image(imgf2, filesize);
  cv::Mat img2_mat = convert_image(img2, 1080, 1920, filesize);
  // cv::Mat img1 = cv::imread("./0030.jpg");
  // cv::Mat img1 = cv::imread("./image0frame0.jpg");
  // cv::Mat img2 = cv::imread("./image0frame30.jpg");
  // if (!img1.data) {
  //   std::cerr << "Error: Could not open or read the image file." << std::endl;
  //   return -1;
  // }

  // cv::Mat img1_mat = cv::imread("./009000.jpg");  

  std::vector<std::vector<float>> results1 = process_4(interpreter,"", "./result0.jpg", nmsrec1, pick1, ids1,  &img1_mat);

  // plotBboxes("./image0frame937.jpg", results1, coco_names, "./output.jpg");

  std::chrono::time_point<std::chrono::system_clock> start1, end1;
  std::chrono::duration<double> elapsed_seconds1;
  start1 = std::chrono::system_clock::now();
  std::vector<std::vector<float>> output_id1=output_id(imgf1, results1);
  end1 = std::chrono::system_clock::now();

  elapsed_seconds1 = end1 - start1;
  printf("output id s: %.10f\n", elapsed_seconds1.count());
  // for (const std::vector<float>& inner_vector : output_id1) {
  //     // Iterate through the inner vector and print each element
  //     for (float value : inner_vector) {
  //         std::cout << value << " ";
  //     }
  //     std::cout << std::endl;  // Print a newline after each inner vector
  // }

  std::vector<std::vector<float>> results2 = process_4(interpreter,"", "./result1.jpg", nmsrec2, pick2, ids2,  &img2_mat);


  // plotBboxes("./image0frame900.jpg", results2, coco_names, "./output2.jpg");

  std::vector<std::vector<float>> output_id2=output_id(imgf2, results2);

  // for (const std::vector<float>& inner_vector : output_id2) {
  //     // Iterate through the inner vector and print each element
  //     for (float value : inner_vector) {
  //         std::cout << value << " ";
  //     }
  //     std::cout << std::endl;  // Print a newline after each inner vector
  // }


  // for (const std::vector<float>& row :results1) {
  //   // Iterate through the elements in each row (inner vector)
  //   for (float element : row) {
  //       std::cout << element << ' ';
  //   }
  //   std::cout << std::endl; // Print a newline after each row
  // }

  // std::cout << "=====" << std::endl;  

  // for (const std::vector<float>& row :results2) {
  //   // Iterate through the elements in each row (inner vector)
  //   for (float element : row) {
  //       std::cout << element << ' ';
  //   }
  //   std::cout << std::endl; // Print a newline after each row
  // }
  // std::cout << "=====" << std::endl; 

  // for (const std::vector<float>& row :output_id1) {
  //   // Iterate through the elements in each row (inner vector)
  //   for (float element : row) {
  //       std::cout << element << ' ';
  //   }
  //   std::cout << std::endl; // Print a newline after each row
  // }

  // std::cout << "=====" << std::endl;  

  // for (const std::vector<float>& row :output_id2) {
  //   // Iterate through the elements in each row (inner vector)
  //   for (float element : row) {
  //       std::cout << element << ' ';
  //   }
  //   std::cout << std::endl; // Print a newline after each row
  // }

  //this part cannot stop running, will test on monday**************
  std::chrono::time_point<std::chrono::system_clock> start2, end2;
  std::chrono::duration<double> elapsed_seconds2;
  start2 = std::chrono::system_clock::now();

  std::vector<std::map<int, std::vector<float>>> compare_result=compare(
      imgf1, results1,output_id1,
      imgf2, results2,output_id2
  );
 
  end2 = std::chrono::system_clock::now();

  elapsed_seconds2 = end2 - start2;
  printf("tracking interpreter s: %.10f\n", elapsed_seconds2.count());

  // for (const auto& map_item : compare_result) {
  //     for (const auto& pair : map_item) {
  //         std::cout << "Key: " << pair.first << std::endl;
  //         std::cout << "Values: ";
  //         for (const auto& value : pair.second) {
  //             std::cout << value << " ";
  //         }
  //         std::cout << std::endl;
  //     }
  // }
  
  for (const auto& pair : compare_result[0]) {
      int key = pair.first;
      const std::vector<float>& values = pair.second;

      std::cout << "Key: " << key << ", Values: ";
      for (const float& value : values) {
          std::cout << value << " ";
      }
      std::cout << std::endl;
  }

  for (const auto& pair : compare_result[1]) {
      int key = pair.first;
      const std::vector<float>& values = pair.second;

      std::cout << "Key: " << key << ", Values: ";
      for (const float& value : values) {
          std::cout << value << " ";
      }
      std::cout << std::endl;
  }

  // std::vector<int> compare_result1;
  // std::vector<int> compare_result2;


  plotBboxes(imgf1, results1, coco_names, "./output.jpg",compare_result[0]);
  plotBboxes(imgf2, results2, coco_names, "./output2.jpg",compare_result[1]);

  std::vector<int> moved_list=motion_detection_pair(results1, compare_result[0], results2, compare_result[1], 80, 0.8);

  for(int i=0;i<moved_list.size();i++){
    std::cout << moved_list[i] << std::endl;
  }


  return 0;


}

