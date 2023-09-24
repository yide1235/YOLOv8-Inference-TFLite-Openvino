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
    // std::cout << boxes.size() << std::endl;

    std::vector<float> x1, y1, x2, y2, areas;
    std::vector<int> indices(boxes.size());

    // Extract coordinates and compute areas

    int a =0;
    while(a<boxes.size()){
        x1.push_back(boxes[a][0]);    
        y1.push_back(boxes[a][1]);
        x2.push_back(boxes[a][2]);
        y2.push_back(boxes[a][3]);
        areas.push_back((x2[a] - x1[a] + 1) * (y2[a] - y1[a] + 1));
        indices[a] = static_cast<int>(a);
        a++;
    }

    std::vector<int> keep;


    std::vector<int> tempIndices;
    int o=0;
    while(o<boxes.size()){
        tempIndices.push_back(o);
        // std::cout << o << std::endl;
        o++;
    }

    
    int q =0;
    while(q<boxes.size()){
    std::vector<int>rest;

        int p=q+1;
        while(p<boxes.size()){
            rest.push_back(p);
            // std::cout << p << std::endl;
            p++;
        }
        // std::cout << q << std::endl;

        // for (int val : rest) {
        //     std::cout << val << " ";
        // }
        // std::cout << std::endl;
        // std::cout << "-------" << std::endl;

        vector<float> xx1;
        vector<float> yy1;
        vector<float> xx2;
        vector<float> yy2;
        int l=q+1;
        while(l<boxes.size()){
            xx1.push_back(std::max(boxes[l][0], boxes[q][0]));
            yy1.push_back(std::max(boxes[l][1], boxes[q][1]));
            xx2.push_back(std::min(boxes[l][2], boxes[q][2]));
            yy2.push_back(std::min(boxes[l][3], boxes[q][3]));

            
            l++;
        }

        assert( xx2.size() == xx1.size());
        
        vector<float>w;
        for(int x=0; x< xx1.size();x++){
            w.push_back(std::max(0.0f,(xx2[x]-xx1[x]+1)));
        }

        assert( yy2.size() == yy1.size());

        vector<float>h;
        for(int y=0; y<yy1.size();y++){
            h.push_back(std::max(0.0f, (yy2[y]-yy1[y]+1)));
        }

        // for(int c=0;c<w.size();c++){
        //     std::cout << w[c] << std::endl;
        // }
        // for(int c=0;c<h.size();c++){
        //     std::cout << h[c] << std::endl;
        // }
        // float xx1 = std::max(boxes[q][0], x1[rest[0]]);
        
        // float yy1 = std::max(boxes[q][1], y1[rest[0]]);
        // float xx2 = std::min(boxes[q][2], x2[rest[0]]);
        // float yy2 = std::min(boxes[q][0], y2[rest[0]]);

        // float w = std::max(0.0f, xx2 - xx1 + 1);
        // float h = std::max(0.0f, yy2 - yy1 + 1);

        vector<float> temp_areas;
        int v=q+1;
        while(v<boxes.size()){
            temp_areas.push_back(areas[v]);
            v++;
        }

        // for(int v=0;v<temp_areas.size();v++){
        //     std::cout << temp_areas[v] << std::endl;
        // }

        vector<float> wxh;
        assert(w.size()==h.size());
        for(int b=0;b<w.size();b++){
            wxh.push_back(w[b]*h[b]);
        }

        vector<float> overlap;
        assert(wxh.size()==temp_areas.size());
        for(int n=0;n<wxh.size();n++){
            overlap.push_back(wxh[n]/temp_areas[n]);
            // std::cout<< wxh[n]/temp_areas[n] << std::endl;
        }

        // float overlap = (w * h) / areas[rest[0]];

        vector<int> results;
        results.push_back(q);
        for(int m=0;m<rest.size();m++){
            results.push_back(rest[m]);
        }

        bool exist=false;
        for(int s=0;s<overlap.size();s++){
            if(overlap[s]>overlapThresh){
                exist=true;
            }
        }
        
        vector<int>temp2;
        // std::cout << exist << "aaaaaaaa" <<std::endl; 
        if(exist==1){
            
            for(int d=0;d<results.size();d++){
                if(results[d]!=q){
                    temp2.push_back(results[d]);
                }
            }
            keep=temp2;
        }
        // std::cout << temp2.size() << "ppppppppppp" <<std::endl; 
        // results=temp2;
        // if (overlap <= overlapThresh) {
        //     for (int index2 : indices) {
        //         if (index2 != i) {
        //             indices.push_back(index2);
        //         }
        //     }
        //     keep.push_back(indices[i]);

        // }
 
        q++;
        // std::cout << "\n" <<std::endl; 
    }
   
    // std::cout << keep[0] <<std::endl; 

    return keep;
}




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

  std::cout << confidence.size()<< std::endl;
  std::cout << index.size()<< std::endl;
  std::cout << bbox.size()<< std::endl;

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



  // std::vector<std::vector<float>> results;




  for(int q=0;q<bbox.size();q++){
      bbox[q]=xywh2xyxy_scale(bbox[q],640,640);
      for(int z=0;z<bbox[q].size();z++){
          // std::cout<< bbox[q][z] << std::endl;

      }
      // std::cout<< "=======" << std::endl;
  }



  std::vector<std::vector<float>> results;

  int confidence_length=confidence.size();
  int class_length=index.size();
  assert(confidence_length == class_length);
  // std::cout << bbox.size()<<std::endl; //11
  // std::cout << bbox[0].size()<<std::endl; //4
  // std::cout << confidence_length<<std::endl;
  assert(confidence_length == bbox.size());

  int j=0; 
  while(j<80){
      std::vector<int> temp;
      std::vector<std::vector<float>>box;
      for (int i=0;i< class_length;i++){

      if(index[i]==j){
          temp.push_back(i);
          box.push_back(bbox[i]);
      }
      }

      std::vector<std::vector<float>> box_selected;
      std::vector<float> confidence_selected;
      
      if (temp.size()>0){

          std::vector<int> indices=NMS(box, 0.45);
          // std::cout << indices.size() <<  std::endl; 

          if(indices.size()>0){
          for(int k=0;k<indices.size();k++){
              box_selected.push_back(box[indices[k]]);
              confidence_selected.push_back(confidence[indices[k]]);
          }
          for(int p=0;p<box_selected.size();p++){
              box_selected[p]=scaleBox(box_selected[p], HEIGHT, WIDTH, static_cast<int>(width), static_cast<int>(height) );
          }

          }
      
      std::vector<float> temp3;
      for(int u=0;u<box_selected.size();u++){
          temp3=box_selected[u];
          temp3.push_back(confidence_selected[u]);
          float floatValue = static_cast<float>(j);
          temp3.push_back(floatValue);
      }
      results.push_back(temp3);
      }
      j++;
  }

  // Iterate through the outer vector
  for (const std::vector<float>& innerVec : results) {
      // Iterate through the inner vector and print its elements
      for (float value : innerVec) {
          std::cout << value << " ";
      }
      std::cout << std::endl; // Print a newline after each inner vector
  }








 //approach two:
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
  //   float max_row=probs.at<float>(0,0);
  //   for(int j=0;j<80;j++){
  //     if(probs.at<float>(i,j)>max_row){
  //       scores2.at<float>(1,i)=probs.at<float>(i,j);
  //       index.at<float>(1,i)=static_cast<float>(j);
  //       max_row=probs.at<float>(i,j);
  //     }
  //   }
  // }
  // std::vector<int>idx_th;
  // for(int i=0;i<scores2.cols;i++){
  //   if(scores2.at<float>(1,i)>0.25f){
  //     idx_th.push_back(i);
  //   }
  // }

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
  // }

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
  //
  //***approach1 and apraoch2 is the same results, but approach1 is way more faster
  //

















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
  
  auto ext_delegate_option = TfLiteExternalDelegateOptionsDefault("/usr/lib/libvx_delegate.so");
  auto ext_delegate_ptr = TfLiteExternalDelegateCreate(&ext_delegate_option);
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  interpreter->ModifyGraphWithDelegate(ext_delegate_ptr);
  interpreter->SetAllowFp16PrecisionForFp32(false);
  interpreter->AllocateTensors();

  std::cout << " Tensorflow Test " << endl;

  // int height =320;
  // int width = 320;
  // string file_name = "image0frame0.jpg";
  // const char* img = read_image(file_name);
  // cout << "Got image char array " << endl;
  // string resp = process_frame(img, height, width);

  // string imgf1= "image0frame40808.jpg";
  // string imgf1 ="8880.jpg";
  string imgf1="image0frame37484.jpg";
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
  cv::Mat img1_mat = convert_image(img1, 1080, 1920, filesize);
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