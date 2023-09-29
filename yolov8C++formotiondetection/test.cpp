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

//one library i added
#include <cmath>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>


using namespace std;
using namespace cv;
using namespace tflite;
// ns
using TfLiteDelegatePtr = tflite::Interpreter::TfLiteDelegatePtr;
// ns
using TfLiteDelegatePtrMap = std::map<std::string, TfLiteDelegatePtr>;

typedef cv::Point3_<float> Pixel;


// std::vector<int> motion_detection_pair(const std::vector<vector<float>>& results1, const std::map<int, std::vector<float>>& ids1,
// const std::vector<vector<float>>& results2, const std::map<int, std::vector<float>>& ids2, int move_threshold, float ratio_threshold){

//   std::vector<int> results;

//   assert(results1.size()==ids1.size());
//   assert(results2.size()==ids2.size());

//   if(results1.size()>results2.size()){
//     //ids1 is the longer one with [0] euqal to index
//     //then traverse ids2
//     // Traverse the map using a range-based for loop
//     for(int i=0;i< results2.size(); i++){
//       auto it= ids2.find(i);
//       if(it!=ids2.end()){
//         const std::vector<float>& values = it->second;//i is the index at ids2
//         int corresp=static_cast<int>(values[0]);//corresp is the index at ids1

//         //now the index is i and corresp, compare the size
//         bool moved=false;
//         if(corresp!=-1){
//           std::vector<float> i_box;//ids2
//           std::vector<float> corresp_box;//ids1

//           for(int m=0;m<4;m++){
//             int each2=static_cast<int>(std::round(results2[i][m]));
//             i_box.push_back(each2);
//           }

//           for(int m=0;m<4;m++){
//             int each2=static_cast<int>(std::round(results1[corresp][m]));
//             corresp_box.push_back(each2);
//           }

//           //check moved
//           int i_x_mid=static_cast<int>(std::round(i_box[2]-i_box[0])/2)+i_box[0];
//           int i_y_mid=static_cast<int>(std::round(i_box[3]-i_box[1])/2)+i_box[1];

//           int corresp_x_mid=static_cast<int>(std::round(corresp_box[2]-corresp_box[0])/2)+corresp_box[0];
//           int corresp_y_mid=static_cast<int>(std::round(corresp_box[3]-corresp_box[1])/2)+corresp_box[1];

          
//           float i_ratio=(i_box[2]-i_box[0])/(i_box[3]-i_box[1]);

//           float corresp_ratio=(corresp_box[2]-corresp_box[0])/(corresp_box[3]-corresp_box[1]);

//           if((i_ratio/corresp_ratio>ratio_threshold)&&((std::abs(corresp_x_mid-i_x_mid)>move_threshold)||(std::abs(corresp_y_mid-i_y_mid)>move_threshold))){
//             moved=true;
//           }
//           // std::cout << std::abs((i_box[2]-i_box[0])*(i_box[3]-i_box[1])) << std::abs((corresp_box[2]-corresp_box[0])*(corresp_box[3]-corresp_box[1])) << std::endl;
//           // assert(1==0);
//           if((0.75> std::abs(std::abs((i_box[2]-i_box[0])*(i_box[3]-i_box[1]))/std::abs((corresp_box[2]-corresp_box[0])*(corresp_box[3]-corresp_box[1]))))||
//           (1.25< std::abs(std::abs((i_box[2]-i_box[0])*(i_box[3]-i_box[1]))/std::abs((corresp_box[2]-corresp_box[0])*(corresp_box[3]-corresp_box[1]))))){
//             moved=false;
//           }
//           std::cout << "class is: "<< " " << corresp << " " << std::abs(corresp_x_mid-i_x_mid) << " " << std::abs(corresp_y_mid-i_y_mid) << " " << i_ratio/corresp_ratio << " "<< std::abs(std::abs((i_box[2]-i_box[0])*(i_box[3]-i_box[1]))/std::abs((corresp_box[2]-corresp_box[0])*(corresp_box[3]-corresp_box[1]))) << std::endl;
//         }

//         if(moved){
//           results.push_back(corresp);//i is the same with corresp
//         }
//       }
//     }

//   }else{
//     //ids2 is [i]==index
//     //travers ids1

//     for(int i=0;i< results1.size(); i++){
//       auto it= ids1.find(i);
//       if(it!=ids1.end()){
//         const std::vector<float>& values = it->second;//i is the index at ids1
//         int corresp=static_cast<int>(values[0]);//corresp is the index at ids2

//         //now the index is i and corresp, compare the size
//         bool moved=false;
//         if(corresp!=-1){
//           std::vector<float> i_box;//ids1
//           std::vector<float> corresp_box;//ids2

//           for(int m=0;m<4;m++){
//             int each2=static_cast<int>(std::round(results1[i][m]));
//             i_box.push_back(each2);
//           }

//           for(int m=0;m<4;m++){
//             int each2=static_cast<int>(std::round(results2[corresp][m]));
//             corresp_box.push_back(each2);
//           }

//           //check moved
//           int i_x_mid=static_cast<int>(std::round(i_box[2]-i_box[0])/2)+i_box[0];
//           int i_y_mid=static_cast<int>(std::round(i_box[3]-i_box[1])/2)+i_box[1];

//           int corresp_x_mid=static_cast<int>(std::round(corresp_box[2]-corresp_box[0])/2)+corresp_box[0];
//           int corresp_y_mid=static_cast<int>(std::round(corresp_box[3]-corresp_box[1])/2)+corresp_box[1];


//           float i_ratio=(i_box[2]-i_box[0])/(i_box[3]-i_box[1]);

//           float corresp_ratio=(corresp_box[2]-corresp_box[0])/(corresp_box[3]-corresp_box[1]);

//           if((i_ratio/corresp_ratio>ratio_threshold)&& ((std::abs(corresp_x_mid-i_x_mid)>move_threshold)||(std::abs(corresp_y_mid-i_y_mid)>move_threshold))){
//             moved=true;
//           }
//           if((0.25> std::abs(std::abs((i_box[2]-i_box[0])*(i_box[3]-i_box[1]))/std::abs((corresp_box[2]-corresp_box[0])*(corresp_box[3]-corresp_box[1]))))||
//           (1.25< std::abs(std::abs((i_box[2]-i_box[0])*(i_box[3]-i_box[1]))/std::abs((corresp_box[2]-corresp_box[0])*(corresp_box[3]-corresp_box[1]))))){
//             moved=false;
//           }
//           std::cout << "class is: "<< " " << corresp << " " << std::abs(corresp_x_mid-i_x_mid) << " " << std::abs(corresp_y_mid-i_y_mid) << " " << i_ratio/corresp_ratio << " "<< std::abs(std::abs((i_box[2]-i_box[0])*(i_box[3]-i_box[1]))/std::abs((corresp_box[2]-corresp_box[0])*(corresp_box[3]-corresp_box[1]))) << std::endl;
//         }

//         if(moved){
//           results.push_back(corresp);//i is the same with corresp
//         }


//       }
//     }
//   }
//   return results;
 
// }




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

      vector<float> temp_areas;
      for(int l=0;l<temp_indices.size();l++){
        temp_areas.push_back(areas[temp_indices[l]]);
      }
 

      vector<float> wxh;
      assert(w.size()==h.size());
      for(int b=0;b<w.size();b++){
          wxh.push_back(w[b]*h[b]);

      }

      vector<float> overlap;
      assert(wxh.size()==temp_areas.size());
      for(int n=0;n<wxh.size();n++){
          overlap.push_back(wxh[n]/temp_areas[n]);
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

    return indices;
}




std::vector<std::vector<float>> process_4(const std::unique_ptr<tflite::Interpreter>& interpreter,const cv::Mat& img)
{



  cout << " Got model " << endl;
  // get input & output layer
  TfLiteTensor *input_tensor = interpreter->tensor(interpreter->inputs()[0]);
  // cout << " Got input " << endl;
  TfLiteTensor *output_box = interpreter->tensor(interpreter->outputs()[0]);
  // cout << " Got output " << endl;
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

  const float width=img.rows;
  const float height=img.cols;

  //here is the same with python
  cv::Mat inputImg = letterbox(img, WIDTH, HEIGHT);


  inputImg = mat_process(inputImg, WIDTH, HEIGHT);

  // interpreter->SetAllowFp16PrecisionForFp32(true);

  start = std::chrono::system_clock::now();
  cout << " GOT INPUT IMAGE " << endl;
  
  // flatten rgb image to input layer.
  // float* input_data = interpreter->typed_input_tensor<float>(0);
  memcpy(input_tensor->data.f, inputImg.ptr<float>(0), HEIGHT*WIDTH*3* sizeof(float));


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
    // cout << "DIM IS " << output_box->dims->data[i] << endl;
    nelem *= output_box->dims->data[i];
  }
  //output is 1x84x8400
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
      if (box_vec[n * 8400 + m + base] >= 0.30) {
        index.push_back(n);
        confidence.push_back(box_vec[n * 8400 + m + base]);


        std::vector<float> temp2;
        int i = 0;
        while (i < 4) {
          temp2.push_back(box_vec[i * 8400 + m]);
          i++;
        }



        bbox.push_back(temp2);
      }
      n++;
    }
    m++;
  }


  for(int q=0;q<bbox.size();q++){
      bbox[q]=xywh2xyxy_scale(bbox[q],640,640);
  }

  std::vector<std::vector<int>> ind;
  for(int i=0;i< 80;i++){
    std::vector<int> temp4;
    for(int j=0;j<index.size();j++){
      if(i==index[j]){

        temp4.push_back(j);
      }
    }

    if(temp4.empty()){
      temp4.push_back(8401);
    }
    ind.push_back(temp4);

  }



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
 
      
      if (ind[i][0]!=8401){
        for(int j=0;j<ind[i].size();j++){
          box_selected.push_back(bbox[ind[i][j]]);
          
        }

        std::vector<int> indices=NMS(box_selected, 0.45);
        if(indices.size()>0){
          for(int s=0;s<indices.size();s++){
            box_afternms.push_back(bbox[ind[i][indices[s]]]);
          }
        }
   
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



  int size_threshold=3872;

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

  return results;

}






//simply copy objects here
std::vector<std::vector<float>> output_id(const cv::Mat& img, const std::vector<std::vector<float>>& results){

    //must assume detected image and results are not 0 in width and height
    std::vector<std::vector<float>> unique_ids;


    int len_results=results.size();

    for (int i =0; i< len_results; ++i){
        int cls_id=results[i][5];
        float confidence = results[i][4];

        vector <float> x;

        for (int j=0; j< 4; ++j){
            int each = static_cast<int>(std::round(results[i][j]));
            x.push_back(each);
        }

        cv::Mat detected=img(cv::Rect(x[0], x[1], x[2] - x[0], x[3] - x[1]));
 

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
            combinedData.push_back(static_cast<float>(value)*5);
        }
        for (const int& value : g_var) {
            combinedData.push_back(static_cast<float>(value)*5);
        }
        for (const int& value : r_var) {
            combinedData.push_back(static_cast<float>(value)*5);
        }


        combinedData.push_back(confidence*100);



        combinedData.push_back(x[0]/3);
        combinedData.push_back(x[1]/3);
        combinedData.push_back(x[2]/3);
        combinedData.push_back(x[3]/3);

        combinedData.push_back((x[2]-x[0])/(x[3]-x[1])*10);
        combinedData.push_back((x[2]-x[0])*3);
        combinedData.push_back((x[3]-x[1])*3);

        combinedData.push_back(static_cast<float>(b_stddev[0]*b_stddev[0]/45));
        combinedData.push_back(static_cast<float>(g_stddev[0]*g_stddev[0]/45));
        combinedData.push_back(static_cast<float>(r_stddev[0]*r_stddev[0]/45));

        unique_ids.push_back(combinedData);

    }//end of for loop


    return unique_ids;

}




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

    // std::cout << detected.rows << detected.cols << std::endl;

    cv::Mat image1 = normalize(detected);
    int height = image1.rows;
    int width = image1.cols;

    // std::cout << height << width << std::endl;

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
    // l4=cv::Mat(1,2, CV_32FC3);
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


    for (int x = 0; x < l1.rows; ++x) {
        // std::cout << l1.at<float>(x, 0) << std::endl;
        sum += pow(pow(l1.at<float>(x, 0) - c1.at<float>(x, 0), 2) +
                    pow(l2.at<float>(x,0) - c2.at<float>(x,0), 2) +
                    pow(l3.at<float>(x,0) - c3.at<float>(x,0), 2),0.5);
        // std::cout << sum << std::endl;
    }


    for (int x = 0; x < l4.rows; ++x) {
        // std::cout << static_cast<float>(l4.at<int>(0, x)) << std::endl;

        mag2 += (pow(static_cast<float>(l4.at<int>(0, x)) - static_cast<float>(c4.at<int>(0, x)), 2));
        // mag2_int+= pow(l4.at<int>(0, x) - c4.at<int>(0, x), 2);
    }

    mag2 = mag2 / pow((static_cast<float>(l4.at<int>(0, 0))  * static_cast<float>(l4.at<int>(0, 1)) )
     + (static_cast<float>(c4.at<int>(0, 0)) * static_cast<float>(c4.at<int>(0, 1))), 0.5);
    float mag1 = pow(sum, 0.5);
    mag2 = pow(mag2, 0.5);

    return mag1 + mag2;
    // return 1;
}


//so now the result is become 4bbox, 1confidence, 1classid, 1lastseen frame, 1trackingid, 1iffindforco, 1co/svd score

void generateIds(std::vector<std::vector<float>>* results) {
    for (int i = 0; i < (*results).size(); ++i) {
      (*results)[i].push_back(0.0);
      (*results)[i].push_back(static_cast<float>(i));
      (*results)[i].push_back(-1.0);
      (*results)[i].push_back(0.0);
    }
    
}

void compare(
    const cv::Mat& img1, std::vector<std::vector<float>>* results1,
    const std::vector<std::vector<float>>& unique_ids1,
    const cv::Mat& img2, std::vector<std::vector<float>>* results2, 
    const std::vector<std::vector<float>>& unique_ids2, float* addition) 
{


  cv::Mat image1 = img1;
  if (image1.empty()) {
      std::cerr << "Failed to load image." << std::endl;
  }

  cv::Mat image2 = img2;
  if (image2.empty()) {
      std::cerr << "Failed to load image." << std::endl;
  }

  // std::cout << "loading image" << std::endl;

  // int svd_threshold = 8;
  // int cut_threshold = 40;


  int svd_threshold=15;
  int cut_threshold=90;



  //now results1 have ids, results2 does not have ids
  //remeber: result is become 4bbox, 1confidence, 1classid, 1lastseen frame, 1trackingid, 1iffindforco, 1updatedco/svd score
  
  // std::cout << "----------------before covariance " << std::endl;
  
  // for (const auto& innerVector : (*results1)) {
  //     // Loop through the inner vector and print its elements
  //     for (const float& value : innerVector) {
  //         std::cout << value << " ";
  //     }
  //     std::cout << std::endl; // Print a newline after each inner vector
  // }

  // std::cout << "----------- " << std::endl;
  // for (const auto& innerVector : (*results2)) {
  //     // Loop through the inner vector and print its elements
  //     for (const float& value : innerVector) {
  //         std::cout << value << " ";
  //     }
  //     std::cout << std::endl; // Print a newline after each inner vector
  // }
  // std::cout << "-----------------after covariance " << std::endl;


  //find by covariance
  for (int i = 0; i < unique_ids2.size(); ++i) {
      float min_norm2 = std::numeric_limits<float>::infinity();
      int matching_id = -1;


      // Compare with vectors in unique_ids2
      for (int j = 0; j < unique_ids1.size(); ++j) {
          if (unique_ids2[i][0] == unique_ids1[j][0]) {//this line ensure they have to be the same class
              float norm2 = 0.0;

              // std::cout << "test position 2" << std::endl;

              for (int k = 1; k < unique_ids2[i].size(); ++k) {

                  norm2 += std::pow(unique_ids1[j][k] - unique_ids2[i][k], 2);
                  // std::cout << "test position 3" << std::endl;
              }
              norm2 = std::sqrt(norm2);

              if (norm2 < min_norm2) {
                  min_norm2 = norm2;
                  matching_id = static_cast<int>(j);
              }
          }
      }
      //                 0,1,2,3,  4,           5,       6,         7,     8,          9
      //result is become 4bbox, 1confidence, 1classid, 1lastseen, 1id, 1iffindforco, 1updatedco/svd score
      if ((cut_threshold > min_norm2)) {
        (*results2)[i].push_back(0.0f); //results[6]lastseen
        (*results2)[i].push_back((*results1)[matching_id][7]);//results[7]id
        (*results2)[i].push_back(1.0f);//result[8]iffindis1, other are -1
        (*results2)[i].push_back(min_norm2);//result[9]updatescore

        //things in results1 should also be change for svd
        (*results1)[matching_id][8]=1;
        //if not find should not change results1
      } 
      else {

          (*results2)[i].push_back(0.0f);//result6
          //note this line could lead error, check at last
          (*results2)[i].push_back(static_cast<float>(std::max((*results1).size(),(*results2).size() ))+(*addition));//result6
          (*addition)++;
          (*results2)[i].push_back(-1.0f);
          (*results2)[i].push_back(0.0f);
      }
      // }

  }
  
  // std::cout << "----------------after covariance " << std::endl;
  
  // for (const auto& innerVector : (*results1)) {
  //     // Loop through the inner vector and print its elements
  //     for (const float& value : innerVector) {
  //         std::cout << value << " ";
  //     }
  //     std::cout << std::endl; // Print a newline after each inner vector
  // }

  // std::cout << "----------- " << std::endl;
  // for (const auto& innerVector : (*results2)) {
  //     // Loop through the inner vector and print its elements
  //     for (const float& value : innerVector) {
  //         std::cout << value << " ";
  //     }
  //     std::cout << std::endl; // Print a newline after each inner vector
  // }
  // std::cout << "-----------------after covariance " << std::endl;



 
  //handle deplicate covariance
  for (int it1 = 0;it1<(*results2).size();it1++) {
    for (int it2 = 0;it2<(*results2).size();it2++) {

      if (it1 != it2 && (*results2)[it1][7] == (*results2)[it2][7]) {
          //make sure they are the same class
        if ((*results2)[it1][5] == (*results2)[it2][5]) {
            //make sure they are all found
          if ((*results2)[it1][8] != -1 && (*results2)[it2][8] != -1) {

            if ((*results2)[it1][9] > (*results2)[it2][9]) {
                (*results2)[it1][7] =  static_cast<float>(std::max((*results1).size(),(*results2).size() ))+(*addition);
                (*addition)++;
            } else {
                (*results2)[it1][7] =  static_cast<float>(std::max((*results1).size(),(*results2).size() ))+(*addition);
                (*addition)++;
            }
          }
        } else {//if they are different class but got the same id, shouldnt happen when assign them
            if ((*results2)[it1][8] != -1 && (*results2)[it2][8] != -1) {
              (*results2)[it1][7] =  static_cast<float>(std::max((*results1).size(),(*results2).size() ))+(*addition);
              (*addition)++;
            }
        }
      }
    }
  }
  
  // std::cout << "------------------------------before svd " << std::endl;
  
  // for (const auto& innerVector : (*results1)) {
  //     // Loop through the inner vector and print its elements
  //     for (const float& value : innerVector) {
  //         std::cout << value << " ";
  //     }
  //     std::cout << std::endl; // Print a newline after each inner vector
  // }

  // std::cout << "----------- " << std::endl;
  // for (const auto& innerVector : (*results2)) {
  //     // Loop through the inner vector and print its elements
  //     for (const float& value : innerVector) {
  //         std::cout << value << " ";
  //     }
  //     std::cout << std::endl; // Print a newline after each inner vector
  // }
  // std::cout << "------------------------------after svd " << std::endl;



  // std::cout << "start to svd" << std::endl;
  //                 0,1,2,3,  4,           5,       6,         7,     8,          9
  //result is become 4bbox, 1confidence, 1classid, 1lastseen, 1id, 1iffindforco, 1updatedco/svd score
  //this is for svd
  for (int i = 0; i < (*results2).size(); ++i) {
      
      if ((*results2)[i][8] == -1.0) {
          vector <float> x;
     
          for (int g=0; g< 4; ++g){
              int each2 = static_cast<int>(std::round((*results2)[i][g]));
              x.push_back(each2);
          }

          cv::Mat detected2 = image2(cv::Rect(x[0], x[1], x[2] - x[0], x[3] - x[1]));

          float class_id2 = (*results2)[i][5];

          int index = -1;
          float min_score = 1000.0;
  

          for (size_t j = 0; j < (*results1).size(); ++j) {

       
                // if ((ids1[j][1] == -1) && (j<results1.size())){
                if (((*results1)[j][8] == -1.0) ){
     
                  vector <float> x2;
                  for (int k=0; k< 4; ++k){
                      int each = static_cast<int>(std::round((*results1)[j][k]));
                      x2.push_back(each);
                  }

        
                  cv::Mat detected1 = image1(cv::Rect(x2[0], x2[1], x2[2] - x2[0], x2[3] - x2[1]));

                  float class_id1 = (*results1)[j][5];


                  if (class_id2 == class_id1){

                    cv::Mat ds_detected2;
                    cv::resize(detected2, ds_detected2, cv::Size(30, 30));

                    
                    std::vector<cv::Mat> l1l4 = calculateSVD(ds_detected2);


                    // std::vector<cv::Mat> l1l4 = calculateSVD(detected2);

         
                    cv::Mat ds_detected1;
                    cv::resize(detected1, ds_detected1, cv::Size(30, 30));
                    std::vector<cv::Mat> c1c4 = calculateSVD(ds_detected1);


                    // std::vector<cv::Mat> c1c4 = calculateSVD(detected1);
   
  
                    cv::Mat l1= l1l4[0];
                    cv::Mat l2= l1l4[1];
                    cv::Mat l3= l1l4[2];
                    cv::Mat l4= l1l4[3];

                    cv::Mat c1=c1c4[0];
                    cv::Mat c2=c1c4[1];
                    cv::Mat c3=c1c4[2];
                    cv::Mat c4=c1c4[3];



                    float ms = get_score(l1,l2,l3,l4,c1,c2,c3,c4);
                    std::cout << ms << std::endl;
                    if (ms < min_score) {
                        min_score = ms;
                        index = j;
                    }
                    
      
                    // std::cout << min_score << std::endl;
           
                    if (min_score < svd_threshold) {
                      (*results2)[i][7]=(*results1)[j][7];
                      (*results2)[i][8]=2.0;
                      (*results2)[i][9]=min_score;
                      (*results1)[j][8]=1.0;
                 
                    }

                  }
              }
          }
      }
  }
  
  // std::cout << "----------------after svd " << std::endl;
  
  // for (const auto& innerVector : (*results1)) {
  //     // Loop through the inner vector and print its elements
  //     for (const float& value : innerVector) {
  //         std::cout << value << " ";
  //     }
  //     std::cout << std::endl; // Print a newline after each inner vector
  // }

  // std::cout << "----------- " << std::endl;
  // for (const auto& innerVector : (*results2)) {
  //     // Loop through the inner vector and print its elements
  //     for (const float& value : innerVector) {
  //         std::cout << value << " ";
  //     }
  //     std::cout << std::endl; // Print a newline after each inner vector
  // }
  // std::cout << "-----------------after svd" << std::endl;


  //now delete duplicate ones for svd
  for(int q=0;q< (*results2).size(); q++){
    if((*results2)[q][8]==2){
      int min_index=q;
      int goal=(*results2)[q][7];
      for(int p=q+1;p<(*results2).size(); p++){
        //so here make sure the value is results8, is both2, also id and class is also same
        if((*results2)[p][8]==2 && (*results2)[p][7]==(*results2)[q][7] && (*results2)[p][5]==(*results2)[q][5]){
          
          if((*results2)[q][9]<=(*results2)[p][9]){
            (*results2)[p][7]=static_cast<float>(std::max((*results1).size(),(*results2).size() ))+(*addition);
            (*addition)++;
            min_index=q;
            //just consider p is the new coming one, but it assigned by svd wrong, so dont need to do anything with the results1[p]
          }else{
            (*results2)[q][0]=static_cast<float>(std::max((*results1).size(),(*results2).size() ))+(*addition);
            (*addition)++;
            min_index=p;

          }
        }
      }
      (*results2)[min_index][7]=goal;
    }
  }

  //add not found in results1 to end of reults2
  for(int i=0;i<(*results1).size();i++){
    if((*results1)[i][8]==-1){
      std::vector<float> temp9=(*results1)[i];
      (*results2).push_back(temp9);
    }
  }

  //if not found, that means it is -1, then set the last seen ++
  //this should be done after draw the boundingbox in the image
  // for(int i=0;i<(*results2).size();i++){
  //   if((*results2)[i][8]==-1){
  //     (*results2)[i][6]++;
  //   }
  // }
  
  std::cout << "-------------------final " << std::endl;
  
  for (const auto& innerVector : (*results1)) {
      // Loop through the inner vector and print its elements
      for (const float& value : innerVector) {
          std::cout << value << " ";
      }
      std::cout << std::endl; // Print a newline after each inner vector
  }

  std::cout << "----------- " << std::endl;
  for (const auto& innerVector : (*results2)) {
      // Loop through the inner vector and print its elements
      for (const float& value : innerVector) {
          std::cout << value << " ";
      }
      std::cout << std::endl; // Print a newline after each inner vector
  }
  std::cout << "--------------------final " << std::endl;

  
}



void update_lastseen(std::vector<std::vector<float>>* results){

  if((*results).size()>0){
    for(int i=0;i<(*results).size();i++){
      if((*results)[i][8]==-1){
        (*results)[i][6]++;
      }
    }
  }

    
}




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
                   const std::string& label = "", int rectLineThickness = 3, int textLineThickness = 2) {
    cv::Point c1(static_cast<int>(x[0]), static_cast<int>(x[1]));
    cv::Point c2(static_cast<int>(x[2]), static_cast<int>(x[3]));
    
    // Draw the rectangle with the specified line thickness
    cv::rectangle(im, c1, c2, color, rectLineThickness, cv::LINE_AA);
    
    if (!label.empty()) {
        int tf = std::max(textLineThickness, 1); // Font thickness
        
        cv::Size textSize = cv::getTextSize(label, 0, 1, tf, nullptr);
        c2 = cv::Point(c1.x + textSize.width, c1.y - textSize.height - 3);
        
        // Draw the filled rectangle for the text background
        cv::rectangle(im, c1, c2, color, -1, cv::LINE_AA);
        
        // Draw the text with the specified line thickness
        cv::putText(im, label, cv::Point(c1.x, c1.y - 2), 0, 1, cv::Scalar(255, 255, 255), tf, cv::LINE_AA);
    }
    
    return im;
}

void plotBboxes(const cv::Mat& img, const std::vector<std::vector<float>>& results,
                const std::vector<std::string>& coco_names, const std::string& savePath) {
    // cv::Mat im0 = cv::imread(imgPath);
    cv::Mat im0=img;
    for (int i = 0; i < results.size(); ++i) {
      if(results[i][6]==0){  //so we print it out whatever if found it or not as long as last seen is 0 
 
        const std::vector<float>& value = results[i];
        std::vector<float> bbox(value.begin(), value.begin() + 4);
        float confidence = value[4];
        int clsId = static_cast<int>(value[5]);
        std::string clsName = coco_names[clsId];

        // Retrieve the tracking ID from the trackingData map
        
        

        int trackingid = static_cast<int>(results[i][7]);

        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << confidence;
        std::string formattedValue = ss.str();
        std::string label = clsName + "" + std::to_string(trackingid) + " " + formattedValue;
        

        cv::Scalar color = getColor(clsId, true);

        im0 = plotOneBox(bbox, im0, color, label);}
    }
    if (!im0.empty()) {
      try {
          cv::imwrite(savePath, im0);
  
      } catch (const std::exception& e) {
          std::cerr << "Error: " << e.what() << std::endl;
      }
    } else {

    }

    // return im0;
}




// // // //this is for inferencing on two images

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

  // string imgf1="./baskcourt_compare/0940.png";
  // string imgf2="./baskcourt_compare/0941.png";  

  // string imgf1="./person_compare/1.jpg";
  // string imgf2="./person_compare/2.jpg"; 

  string imgf1="./bus_compare2/1.jpg";
  string imgf2="./bus_compare2/2.jpg";
  string imgf3="./bus_compare2/3.jpg";
  string imgf4="./bus_compare2/4.jpg";

  // string imgf1="./bus_compare/0030.jpg";
  // string imgf2="./bus_compare/0060.jpg";
  // string imgf3="./bus_compare/0090.jpg";
  
  // string imgf1="./house_compare/l1.png";
  // string imgf2="./house_compare/l2.png";
  // string imgf3="./house_compare/l3.png";
  // string imgf4="./house_compare/l4.png";
  // string imgf5="./house_compare/l5.png";
  // string imgf6="./house_compare/l6.png";
  // string imgf7="./house_compare/l7.png";
  // string imgf8="./house_compare/l8.png";
  // string imgf9="./house_compare/l9.png";
  // string imgf10="./house_compare/l10.png";


  //good
  // string imgf1="./s1.jpg";
  // string imgf2="./s2.jpg";
  // string imgf3="./s3.jpg";
  // string imgf4="./s4.jpg";


  std::chrono::time_point<std::chrono::system_clock> beg, start, end, done, nmsdone;
  std::chrono::duration<double> elapsed_seconds;

  cv::Mat img1 = cv::imread(imgf1);
  if (img1.empty()) {
      std::cerr << "Failed to load image." << std::endl;
      // You should return an empty cv::Mat or handle errors differently.
      return 0;
  }

  cv::Mat img2 = cv::imread(imgf2);
  if (img2.empty()) {
      std::cerr << "Failed to load image." << std::endl;
      // You should return an empty cv::Mat or handle errors differently.
      return 0;
  }

  cv::Mat img3 = cv::imread(imgf3);
  if (img3.empty()) {
      std::cerr << "Failed to load image." << std::endl;
      // You should return an empty cv::Mat or handle errors differently.
      return 0;
  }



  cv::Mat img4 = cv::imread(imgf4);
  if (img4.empty()) {
      std::cerr << "Failed to load image." << std::endl;
      // You should return an empty cv::Mat or handle errors differently.
      return 0;
  }
  
  // cv::Mat img5 = cv::imread(imgf5);
  // if (img5.empty()) {
  //     std::cerr << "Failed to load image." << std::endl;
  //     // You should return an empty cv::Mat or handle errors differently.
  //     return 0;
  // }

  // cv::Mat img6 = cv::imread(imgf6);
  // if (img6.empty()) {
  //     std::cerr << "Failed to load image." << std::endl;
  //     // You should return an empty cv::Mat or handle errors differently.
  //     return 0;
  // }

  // cv::Mat img7 = cv::imread(imgf7);
  // if (img7.empty()) {
  //     std::cerr << "Failed to load image." << std::endl;
  //     // You should return an empty cv::Mat or handle errors differently.
  //     return 0;
  // }

  // cv::Mat img8 = cv::imread(imgf8);
  // if (img8.empty()) {
  //     std::cerr << "Failed to load image." << std::endl;
  //     // You should return an empty cv::Mat or handle errors differently.
  //     return 0;
  // }

  // cv::Mat img9 = cv::imread(imgf9);
  // if (img9.empty()) {
  //     std::cerr << "Failed to load image." << std::endl;
  //     // You should return an empty cv::Mat or handle errors differently.
  //     return 0;
  // }

  // cv::Mat img10 = cv::imread(imgf10);
  // if (img10.empty()) {
  //     std::cerr << "Failed to load image." << std::endl;
  //     // You should return an empty cv::Mat or handle errors differently.
  //     return 0;
  // }

  float addition=1;


  std::vector<std::vector<float>> results1 = process_4(interpreter,img1);

 
  std::vector<std::vector<float>> output_id1=output_id(img1, results1);


  // for(int i=0;i<results1.size();i++){
  //   results1[i].push_back(0.0);
  // }
  



  //so now the result is become 4bbox, 1confidence, 1classid, 1lastseen frame, 1trackingid, 
  generateIds(&results1);


  plotBboxes(img1, results1, coco_names, "./output1.jpg");
  update_lastseen(&results1);




  std::vector<std::vector<float>> results2 = process_4(interpreter,img2);

 
  std::vector<std::vector<float>> output_id2=output_id(img2, results2);

  start = std::chrono::system_clock::now();
  compare(
      img1, &results1,output_id1,
      img2, &results2,output_id2, &addition
  );
  end = std::chrono::system_clock::now();
  elapsed_seconds = end - start;
  printf("The time for compare was s: %.10f\n", elapsed_seconds.count());
  output_id2=output_id(img2, results2);

  

  plotBboxes(img2, results2, coco_names, "./output2.jpg");

  update_lastseen(&results2);





  std::vector<std::vector<float>> results3 = process_4(interpreter,img3);

 
  std::vector<std::vector<float>> output_id3=output_id(img3, results3);



  start = std::chrono::system_clock::now();
  compare(
      img2, &results2,output_id2,
      img3, &results3,output_id3, &addition
  );
  end = std::chrono::system_clock::now();
  elapsed_seconds = end - start;
  printf("The time for compare was s: %.10f\n", elapsed_seconds.count());



  output_id3=output_id(img3, results3);

  plotBboxes(img3, results3, coco_names, "./output3.jpg");

  update_lastseen(&results3);









  std::vector<std::vector<float>> results4 = process_4(interpreter,img4);

 
  std::vector<std::vector<float>> output_id4=output_id(img4, results4);


  start = std::chrono::system_clock::now();
  compare(
      img3, &results3,output_id3,
      img4, &results4,output_id4, &addition
  );
  end = std::chrono::system_clock::now();
  elapsed_seconds = end - start;
  printf("The time for compare was s: %.10f\n", elapsed_seconds.count());

  output_id4=output_id(img4, results4);

  plotBboxes(img4, results4, coco_names, "./output4.jpg");

  update_lastseen(&results4);







  // std::vector<std::vector<float>> results5 = process_4(interpreter,img5);

 
  // std::vector<std::vector<float>> output_id5=output_id(img5, results5);
  // start = std::chrono::system_clock::now();
  // compare(
  //     img4, &results4,output_id4,
  //     img5, &results5,output_id5, &addition
  // );
  // end = std::chrono::system_clock::now();
  // elapsed_seconds = end - start;
  // printf("The time for compare was s: %.10f\n", elapsed_seconds.count());


  // output_id5=output_id(img5, results5);

  // plotBboxes(img5, results5, coco_names, "./output5.jpg");







  // std::vector<std::vector<float>> results6 = process_4(interpreter,img6);

 
  // std::vector<std::vector<float>> output_id6=output_id(img6, results6);
  // start = std::chrono::system_clock::now();
  // compare(
  //     img5, &results5,output_id5,
  //     img6, &results6,output_id6, &addition
  // );
  // end = std::chrono::system_clock::now();
  // elapsed_seconds = end - start;
  // printf("The time for compare was s: %.10f\n", elapsed_seconds.count());


  // output_id6=output_id(img6, results6);

  // plotBboxes(img6, results6, coco_names, "./output6.jpg");






  // std::vector<std::vector<float>> results7 = process_4(interpreter,img7);

 
  // std::vector<std::vector<float>> output_id7=output_id(img7, results7);
  // start = std::chrono::system_clock::now();
  // compare(
  //     img6, &results6,output_id6,
  //     img7, &results7,output_id7, &addition
  // );
  // end = std::chrono::system_clock::now();
  // elapsed_seconds = end - start;
  // printf("The time for compare was s: %.10f\n", elapsed_seconds.count());


  // output_id7=output_id(img7, results7);

  // plotBboxes(img7, results7, coco_names, "./output7.jpg");







  // std::vector<std::vector<float>> results8 = process_4(interpreter,img8);

 
  // std::vector<std::vector<float>> output_id8=output_id(img8, results8);
  // start = std::chrono::system_clock::now();
  // compare(
  //     img7, &results7,output_id7,
  //     img8, &results8,output_id8, &addition
  // );
  // end = std::chrono::system_clock::now();
  // elapsed_seconds = end - start;
  // printf("The time for compare was s: %.10f\n", elapsed_seconds.count());



  // output_id8=output_id(img8, results8);

  // plotBboxes(img8, results8, coco_names, "./output8.jpg");







  // std::vector<std::vector<float>> results9 = process_4(interpreter,img9);

 
  // std::vector<std::vector<float>> output_id9=output_id(img9, results9);
  // start = std::chrono::system_clock::now();
  // compare(
  //     img8, &results8,output_id8,
  //     img9, &results9,output_id9, &addition
  // );
  // end = std::chrono::system_clock::now();
  // elapsed_seconds = end - start;
  // printf("The time for compare was s: %.10f\n", elapsed_seconds.count());



  // output_id9=output_id(img9, results9);

  // plotBboxes(img9, results9, coco_names, "./output9.jpg");








  // std::vector<std::vector<float>> results10 = process_4(interpreter,img10);

 
  // std::vector<std::vector<float>> output_id10=output_id(img10, results10);
  // start = std::chrono::system_clock::now();
  // compare(
  //     img9, &results9,output_id9,
  //     img10, &results10,output_id10, &addition
  // );
  // end = std::chrono::system_clock::now();
  // elapsed_seconds = end - start;
  // printf("The time for compare was s: %.10f\n", elapsed_seconds.count());



  // output_id10=output_id(img10, results10);

  // plotBboxes(img10, results10, coco_names, "./output10.jpg");




//   // std::vector<int> moved_list=motion_detection_pair(results1, compare_result[0], results2, compare_result[1], 80, 0.80);
//   // // std::vector<int> moved_list=motion_detection_pair(results1, compare_result[0], results2, compare_result[1], 10, 0.70);

//   // for(int i=0;i<moved_list.size();i++){
//   //   std::cout << moved_list[i] << std::endl;
//   // }


  return 0;


}









// // // // this is for inferencing on video:
// int main(int argc, char **argv)
// {

//   std::vector<std::string> coco_names = {
//       "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
//       "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
//       "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
//       "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
//       "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
//       "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
//       "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
//       "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
//       "teddy bear", "hair drier", "toothbrush"
//   };


//     // create model
//   std::unique_ptr<tflite::FlatBufferModel> model =
//       tflite::FlatBufferModel::BuildFromFile("yolov8l_integer_quant.tflite");
  
//   auto ext_delegate_option = TfLiteExternalDelegateOptionsDefault("/usr/lib/libvx_delegate.so");
//   auto ext_delegate_ptr = TfLiteExternalDelegateCreate(&ext_delegate_option);
//   tflite::ops::builtin::BuiltinOpResolver resolver;
//   std::unique_ptr<tflite::Interpreter> interpreter;
//   tflite::InterpreterBuilder(*model, resolver)(&interpreter);
//   interpreter->ModifyGraphWithDelegate(ext_delegate_ptr);
//   interpreter->SetAllowFp16PrecisionForFp32(false);
//   interpreter->AllocateTensors();

//   std::cout << " Tensorflow Test " << endl;


//   cv::VideoCapture cap("./11s.mp4"); 
//   if (!cap.isOpened()) {
//       std::cerr << "Error: Couldn't open input video!" << std::endl;
//       return -1;
//   }

//   int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
//   int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

//   // cv::VideoWriter output_video("./output.mp4", cv::VideoWriter::fourcc('X', '2', '6', '4'), 25, cv::Size(frame_width, frame_height));
//   // cv::VideoWriter output_video("./output_video.mp4", cv::VideoWriter::fourcc('M', 'P', '4', 'V'), 25, cv::Size(frame_width, frame_height));
//   // cv::VideoWriter video("output.avi",CV_FOURCC('M','J','P','G'),25, Size(frame_width,frame_height));
//   cv::VideoWriter video("output_11s.avi", cv::VideoWriter::fourcc('M','J','P','G'), 25, Size(frame_width,frame_height));

//   if (!video.isOpened()) {
//       std::cerr << "Error: Couldn't create output video!" << std::endl;
//       return -1;
//   }

//   cv::Mat prev_frame, current_frame;
//   cv::Mat result_frame; // Store the result frame with bounding boxes

//   cap >> current_frame;
//   std::vector<std::vector<float>> first1 = process_4(interpreter,current_frame);


//   std::vector<std::vector<float>> first_id1=output_id(current_frame, first1);
//   std::map<int, std::vector<float>> ids1=generateIds(first_id1);


//   float addition=20;

//   while (true) {
    
//     prev_frame = current_frame.clone();

//     cap >> current_frame;
    

//     if (current_frame.empty()) {
//         break; // End of video
//     }

//     if (!prev_frame.empty()) {

    

//       std::vector<std::vector<float>> results1 = process_4(interpreter,prev_frame);


//       std::vector<std::vector<float>> output_id1=output_id(prev_frame, results1);

//       // std::cout <<"finished first one" << std::endl;

//       // for (const std::vector<float>& row : output_id1) {
//       //     // Iterate through the elements in each row
//       //     for (float element : row) {
//       //         std::cout << element << " ";
//       //     }
//       //     // Add a newline after each row
//       //     std::cout << std::endl;
//       // }

//       std::vector<std::vector<float>> results2 = process_4(interpreter,current_frame);

     

//       std::vector<std::vector<float>> output_id2=output_id(current_frame, results2);

//       // std::cout <<"finished second one" << std::endl;

//       // for (const std::vector<float>& row : output_id2) {
//       //     // Iterate through the elements in each row
//       //     for (float element : row) {
//       //         std::cout << element << " ";
//       //     }
//       //     // Add a newline after each row
//       //     std::cout << std::endl;
//       // }

//       std::map<int, std::vector<float>> ids2=compare(
//           prev_frame, results1,output_id1,
//           current_frame, results2,output_id2, ids1,  &addition
//       );
//       // std::cout << "after compare" << std::endl;
//       // cv::Mat prev_result_frame=plotBboxes(prev_frame, results1, coco_names, "./output.jpg",compare_result[0]);

//       cv::Mat current_result_frame=plotBboxes(current_frame, results2, coco_names, "./output2.jpg",ids2);
//       // Write the original frames to the output video
//       // video.write(prev_result_frame);
//       video.write(current_result_frame);

//       ids1=ids2;

      
//     }

//     // Break the loop if the user presses 'q'
//     if (cv::waitKey(1) == 'q') {
//         break;
//     }

//   }
//   // When everything done, release the video capture and write object
//   cap.release();
//   video.release();
 
//   // Closes all the frames
//   destroyAllWindows();


//   // for(int i=0;i<moved_list.size();i++){
//   //   std::cout << moved_list[i] << std::endl;
//   // }


//   return 0;


// }



