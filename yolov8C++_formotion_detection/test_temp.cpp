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


std::vector<int> motion_detection_pair(const std::vector<vector<float>>& results1, 
const std::vector<vector<float>>& results2, int move_threshold, float ratio_threshold){
  std::vector<int> results;
  for(int i=0;i< results1.size(); i++){
    for(int j=0;j<results2.size();j++){
      //                 0,1,2,3,  4,           5,       6,         7,     8,          9
      //result is become 4bbox, 1confidence, 1classid, 1lastseen, 1id, 1iffindforco, 1updatedco/svd score
      bool moved=false;
      if((results1[i][7]==results2[j][7])&&(results1[i][8]!=-1)&&(results2[j][8]!=-1)){

        std::vector<float> i_box;//ids2
        std::vector<float> corresp_box;//ids1

        for(int m=0;m<4;m++){
          int each2=static_cast<int>(std::round(results2[j][m]));
          i_box.push_back(each2);
        }

        for(int m=0;m<4;m++){
          int each2=static_cast<int>(std::round(results1[i][m]));
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
        std::cout << i_ratio/corresp_ratio << std::abs(corresp_x_mid-i_x_mid) << std::abs(corresp_y_mid-i_y_mid) << std::endl;
        // assert(1==0);
        // if((0.75> std::abs(std::abs((i_box[2]-i_box[0])*(i_box[3]-i_box[1]))/std::abs((corresp_box[2]-corresp_box[0])*(corresp_box[3]-corresp_box[1]))))||
        // (1.25< std::abs(std::abs((i_box[2]-i_box[0])*(i_box[3]-i_box[1]))/std::abs((corresp_box[2]-corresp_box[0])*(corresp_box[3]-corresp_box[1]))))){
        //   moved=false;
        // }
        // std::cout << "trackingis is: "<< " " << i << " " << j << std::abs(corresp_x_mid-i_x_mid) << " " << std::abs(corresp_y_mid-i_y_mid) << " " << i_ratio/corresp_ratio << " "<< std::abs(std::abs((i_box[2]-i_box[0])*(i_box[3]-i_box[1]))/std::abs((corresp_box[2]-corresp_box[0])*(corresp_box[3]-corresp_box[1]))) << std::endl;
      }

      if(moved){
        results.push_back(results2[j][7]);//i is the same with corresp
      }
    }
    
    
  }

  return results;
 
}




auto mat_process(cv::Mat src, uint width, uint height) -> cv::Mat
{
  // convert to float; BGR -> RGB
  cv::Mat dst2;
  std::cout << "Creating dst" << endl;
  // src.convertTo(dst, CV_32FC3);
  std::cout << "Creating dst2" << endl;
  cv::cvtColor(src, dst2, cv::COLOR_BGR2RGB);
  std::cout << "Creating dst3" << endl;


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



  std::cout << " Got model " << endl;
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
  std::cout << " GOT INPUT IMAGE " << endl;
  
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
      if (box_vec[n * 8400 + m + base] >= 0.45) {
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



  // int size_threshold=3872;
  int size_threshold=3400;

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
std::vector<std::vector<float>> output_id(const cv::Mat& img, std::vector<std::vector<float>>& results){

    // cv::Mat img = cv::imread(img_path);
    // if (img.empty()) {
    //     std::cerr << "Failed to load image." << std::endl;
    //     // You should return an empty cv::Mat or handle errors differently.
    //     return cv::Mat();
    // }

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


    // std::cout << mag2<< std::endl;

    mag2 = mag2 / pow((static_cast<float>(l4.at<int>(0, 0))  * static_cast<float>(l4.at<int>(0, 1)) )
     + (static_cast<float>(c4.at<int>(0, 0)) * static_cast<float>(c4.at<int>(0, 1))), 0.5);
    // std::cout << mag2 << std::endl;
    float mag1 = pow(sum, 0.5);
    mag2 = pow(mag2, 0.5);

    // std::cout << mag1 << std::endl;
    // std::cout << mag2 << std::endl;
    return mag1 + mag2;
    // return 1;
}



//so now the result is become 4bbox, 1confidence, 1classid, 1lastseen frame, 1trackingid, 1iffindforco, 1co/svd score

void generateIds(std::vector<std::vector<float>>* results, std::vector<int>*id_list, std::vector<float>* myList) {
    for (int i = 0; i < (*results).size(); ++i) {
      (*results)[i].push_back(0.0);
      (*results)[i].push_back(static_cast<float>(i));
      
      auto it = std::find((*myList).begin(), (*myList).end(), (*results)[i][5]);

      if (it != (*myList).end()) {
          (*id_list).push_back(i);
          std::cout << i << std::endl;
      } else {
          //do nothing
      }

      (*results)[i].push_back(-1.0);
      (*results)[i].push_back(0.0);

      
    }

    
    
}

void compare(
    const cv::Mat& img1, std::vector<std::vector<float>>* results1,
    const std::vector<std::vector<float>>& unique_ids1,
    const cv::Mat& img2, std::vector<std::vector<float>>* results2, 
    const std::vector<std::vector<float>>& unique_ids2, float* addition, std::vector<int>* id_list,
    const std::vector<float>& myList
    ) 
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


  // int svd_threshold=40;
  // int cut_threshold=120;

  int svd_threshold=80;
  int cut_threshold=180;


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
          auto it = std::find(myList.begin(), myList.end(), (*results2)[i][5]);
          if(it != myList.end()) {
            (*id_list).push_back(static_cast<float>(std::max((*results1).size(),(*results2).size() ))+(*addition));}
          
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



 
  // handle deplicate covariance
  for (int it1 = 0;it1<(*results2).size();it1++) {
    for (int it2 = 0;it2<(*results2).size();it2++) {

      if (it1 != it2 && (*results2)[it1][7] == (*results2)[it2][7]) {
          //make sure they are the same class
        if ((*results2)[it1][5] == (*results2)[it2][5]) {
            //make sure they are all found
          if ((*results2)[it1][8] != -1 && (*results2)[it2][8] != -1) {

            if ((*results2)[it1][9] > (*results2)[it2][9]) {
                (*results2)[it1][7] =  static_cast<float>(std::max((*results1).size(),(*results2).size() ))+(*addition);
                auto it = std::find(myList.begin(), myList.end(), (*results2)[it1][5]);
                if(it != myList.end()){
                  (*id_list).push_back(static_cast<float>(std::max((*results1).size(),(*results2).size() ))+(*addition));
                }
                
                (*addition)++;
            } else {
                (*results2)[it2][7] =  static_cast<float>(std::max((*results1).size(),(*results2).size() ))+(*addition);
                auto it = std::find(myList.begin(), myList.end(), (*results2)[it2][5]);
                if(it != myList.end()){
                  (*id_list).push_back(static_cast<float>(std::max((*results1).size(),(*results2).size() ))+(*addition));
                }
                
                (*addition)++;
            }
          }
        } else {//if they are different class but got the same id, shouldnt happen when assign them
            if ((*results2)[it1][8] != -1 && (*results2)[it2][8] != -1) {
              (*results2)[it1][7] =  static_cast<float>(std::max((*results1).size(),(*results2).size() ))+(*addition);
              auto it = std::find(myList.begin(), myList.end(), (*results2)[it1][5]);
              if(it != myList.end()){
                (*id_list).push_back(static_cast<float>(std::max((*results1).size(),(*results2).size() ))+(*addition));
              }
              
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
                    cv::resize(detected2, ds_detected2, cv::Size(200, 200));

                    
                    std::vector<cv::Mat> l1l4 = calculateSVD(ds_detected2);


                    // std::vector<cv::Mat> l1l4 = calculateSVD(detected2);

         
                    cv::Mat ds_detected1;
                    cv::resize(detected1, ds_detected1, cv::Size(200, 200));
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
            auto it = std::find(myList.begin(), myList.end(), (*results2)[p][5]);
            if(it != myList.end()){
              (*id_list).push_back(static_cast<float>(std::max((*results1).size(),(*results2).size() ))+(*addition));
            }
            
            (*addition)++;
            min_index=q;
            //just consider p is the new coming one, but it assigned by svd wrong, so dont need to do anything with the results1[p]
          }else{
            (*results2)[q][0]=static_cast<float>(std::max((*results1).size(),(*results2).size() ))+(*addition);
            auto it = std::find(myList.begin(), myList.end(), (*results2)[q][5]);
            if(it != myList.end()){
              (*id_list).push_back(static_cast<float>(std::max((*results1).size(),(*results2).size() ))+(*addition));
            }
            
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
  
  // std::cout << "-------------------final " << std::endl;
  
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
  // std::cout << "--------------------final " << std::endl;

  
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
                const std::vector<std::string>& coco_names, const std::string& savePath, const std::vector<int>& id_list) {
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

        auto it = std::find(id_list.begin(), id_list.end(), trackingid);
        std::string label;
        if (it != id_list.end()) {
            // Element found
            label = clsName + "" + std::to_string(trackingid) + "motion" + formattedValue;
        } else {
            // Element not found
            label = clsName + "" + std::to_string(trackingid) + " " + formattedValue;
        }
        
        

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





 // this is for inferencing on video, two frames per second
int get_next_video(){ //update this reference
    return 30;

}

string get_video_string(int seq){ //update this reference
  return "./" + to_string(seq) + "min.mp4";

}

bool keep_ml_thread_alive(){  //update this reference
    return true;
}

vector<string> check_zones(cv::Rect test_rec){ //update this reference
  
  vector<string> zone_intersections;
  zone_intersections.push_back("Z1");
  zone_intersections.push_back("Z2");
  zone_intersections.push_back("Z3");
  zone_intersections.push_back("Z4");
  return zone_intersections;
}

string get_meta_data(cv::Mat mat1, std::vector<std::vector<float>> results){
   
  vector<cv::Rect> nmsrec; 
  //get vector of rectangles   xmin, ymin, xmax-xmin, ymax-ymin
  //                 0,1,2,3,  4,           5,       6,         7,     8,          9
  //result is become 4bbox, 1confidence, 1classid, 1lastseen, 1id, 1iffindforco, 1updatedco/svd score
  for(int cnt=0; cnt<results.size(); cnt++){
    int xmin = results[cnt][0];
    int ymin = results[cnt][1];
    int xmax = results[cnt][2];
    int ymax = results[cnt][3];
    nmsrec.push_back(cv::Rect(xmin,ymin,xmax-xmin, ymax-ymin));
  }


  string objects = "";
  string bndbox = ""; 
  string bndzones = "";
  string zoneclassstr = "";
  map<string,map<int,int>> zone_classes;
  //string zonedef="";

  int pick_index = 0;
  //is_motion = false;
  for(cv::Rect rect: nmsrec){
    //cout << " Testing rect as pick index " << pick_index << " which has class id " << ids[pick[pick_index]] << endl;
    //std::cout << " Cut off value is " << to_string(eval[pick_index] ) << " and motion cut off is " << to_string(MOTION_CUTOFF) << " which has class id " << ids[pick[pick_index]] << endl;
    /*if(eval[pick_index] > MOTION_CUTOFF){
      if(!check_motion_setting(ids[pick[pick_index]])){
        #ifdef VIPER_DEBUG_ENABLED
          std::cout << " Ignoring this motion due to suppressed class " << endl;
        #endif
        continue;
      }//if global motion settings are turned off ignore motion from this class of object
      #ifdef VIPER_DEBUG_ENABLED
        std::cout << " Motion value is " << is_motion << endl;
      #endif
      is_motion=true; //set the flag that there was motion
      motion.push_back(pick_index);
      #ifdef VIPER_DEBUG_ENABLED
        std::cout << "Motion cutoff value is " << eval[pick_index] << " motion cutoff is " << MOTION_CUTOFF << " total stddev is " << total_stdev  << endl;
      #endif
    */
      //cout << " Rect as pick index " << pick_index << " shows motion with class id " << ids[pick[pick_index]] << endl;
    

      int curr_class = results[pick_index][5];
      cv::rectangle(mat1, rect, cv::Scalar(0, 255, 0), 3);
      cv::putText(mat1, to_string(curr_class), cv::Point(rect.x+0.5 * rect.width, rect.y+0.5*rect.height), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(118,185,0),2);
      objects += to_string(curr_class) + ","; //to_string(ids[pick[pick_index]]) + ",";
      bndbox += "[" + to_string(rect.x) + "," + to_string(rect.y) + "," + to_string(rect.x+rect.width) + "," + to_string(rect.y + rect.height) + "],"; 

      vector<string> zone_result = check_zones(rect);
      bndzones += "[";
      for(int test_res=0; test_res<zone_result.size(); test_res++){
          bndzones += zone_result.at(test_res) + ",";
          if(zone_classes.find(zone_result.at(test_res)) == zone_classes.end()){
            map<int,int> mp;
            mp[curr_class]=1;
            //#ifdef VIPER_DEBUG_ENABLED
            //  std::cout << " Set the zone class " << ids[pick[pick_index]] << " in the zone " << zone_result.at(test_res) << endl;
            //#endif
            zone_classes[zone_result.at(test_res)] = mp;
          }else{
            zone_classes[zone_result.at(test_res)][curr_class]=1;
          }
      }
      bndzones = bndzones.substr(0,bndzones.length()-1);
      bndzones += "],";
   

    //}
    pick_index++;
  }

  for(const auto &zone_pair : zone_classes){
    map<int,int> zone_class_map = zone_pair.second;
    string zone_name = zone_pair.first;
    string zone_val = "[" + zone_name + ":";
    for(const auto &map_pair : zone_class_map){
      //#ifdef VIPER_DEBUG_ENABLED
      //  std::cout << " The zone map has " << to_string(map_pair.first) << endl;
      //#endif
      zone_val += to_string(map_pair.first) + ",";
    } 
    zone_val = zone_val.substr(0,zone_val.length()-1) + "],";
    zoneclassstr += zone_val;

  }

  objects = "<rt7:Object>[" + objects.substr(0,objects.length()-1) + "]</rt7:Object>";
  bndbox =  "<rt7:BndBox>[" + bndbox.substr(0,bndbox.length()-1) + "]</rt7:BndBox>";
  bndzones = "<rt7:ObjectZone>[" + bndzones.substr(0,bndzones.length()-1) + "]</rt7:ObjectZone>";
  zoneclassstr = "<rt7:ZoneClasses>[" + zoneclassstr.substr(0,zoneclassstr.length()-1)  + "]</rt7:ZoneClasses>";
  
  //add the image reference and the zone classes
  //add zone intersections here
  //#ifdef VIPER_DEBUG_ENABLED
  //  std::cout << " Computed abs diff " << endl;
    
    //imwrite(image_name,mat2);
  //  std::cout << " Wrote diff image " << endl;
  //#endif
  /*
    opencv absdiff get max change
    compare the max frame
    check absdiff of objects in max frame
    determine motion
    develop motion metadata
  */

  return objects + bndbox + bndzones + zoneclassstr;


}

string datetm_to_str(std::tm localtime){  //update this reference  //delete it exists in lib_str.cpp
    try{
		cout << " Got time ................. " << endl;
        char buffer[20];
        std::strftime(buffer,sizeof(buffer), "%m/%d/%Y %H:%M:%S", &localtime);
        string dt(buffer);
        return dt;
    }catch(Exception e){
        cout << "Exception in datetm_to_str " << e.what() << endl;
        return "";
    }
}



std::vector<std::vector<float>> results2motion(std::vector<std::vector<float>>* results, std::vector<int>* id_list){

  std::vector<std::vector<float>> motion_results;

  for (int i = 0; i < (*results).size(); ++i) {
    int temp=static_cast<int>((*results)[i][7]);

    auto it = std::find((*id_list).begin(), (*id_list).end(), temp);
    if (it != (*id_list).end()) {
      motion_results.push_back((*results)[i]);
    } 
  }  

  return motion_results;

}




void remove_unfind(std::vector<std::vector<float>>* results) {
    float remove_threshold = 10.0;

    // Use remove_if along with erase to remove elements based on the condition
    (*results).erase(std::remove_if((*results).begin(), (*results).end(), [=](const std::vector<float>& innerVec) {
        // Assuming innerVec has at least 9 elements (0-based index)
        return innerVec.size() > 6 && innerVec[8] == -1.0 && innerVec[6] > remove_threshold;
    }), (*results).end());
}







int ml_processing_thread()
{

  string std_videos_save_dir = "./"; //update this reference  //delete this it is a global var
  string std_video_image_extension = ".jpg"; //update this reference  //delete this it is a global var
  const int std_videos_jpeg_quality = 37; //update this reference  //delete this it is a global var
  int segment_id;
  string std_video_metadata_extension = ".meta"; //update this reference  //delete this it is a global var





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
  //motion object: 0,1,2,3,5,7
  // std::vector<float> myList={0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  std::vector<float> myList={0.0};
  // create model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile("./yolov8l_integer_quant.tflite");
  
  auto ext_delegate_option = TfLiteExternalDelegateOptionsDefault("/usr/lib/libvx_delegate.so");
  auto ext_delegate_ptr = TfLiteExternalDelegateCreate(&ext_delegate_option);
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  interpreter->ModifyGraphWithDelegate(ext_delegate_ptr);
  interpreter->SetAllowFp16PrecisionForFp32(false);
  interpreter->AllocateTensors();

  while(keep_ml_thread_alive()){
    cv::VideoCapture cap("./4s.mp4");
    // segment_id = get_next_video();
    // cv::VideoCapture cap(get_video_string(segment_id));

    if (!cap.isOpened()) {
        std::cerr << "Error: Couldn't open input video!" << std::endl;
        continue;
    }
    // int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    // int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS)); //video frame rate
    float addition=1; //increment for the new id
    //float drop_threshold=20; //
    std::chrono::time_point<std::chrono::system_clock> beg, start, end, done, nmsdone;
    std::chrono::duration<double> elapsed_seconds;
    std::vector<std::vector<float>> results1;
    //results1 and results2 are vector of vectors of the following type
    //                 0,1,2,3,  4,           5,       6,         7,     8,          9
    //result is become 4bbox, 1confidence, 1classid, 1lastseen, 1id, 1iffindforco, 1updatedco/svd score

    //output_id is the same as results1 4bbox, conf, classid without the rest 
    std::vector<std::vector<float>> output_id1;
    std::vector<std::vector<float>> results2;
    std::vector<std::vector<float>> output_id2;
    std::vector<int> id_list; //object_ids 
    std::vector<int> moved_list;
    cv::Mat last_frame;
    cv::Mat end_frame;
    std::vector<std::vector<float>> motion_results2;
    std::vector<std::vector<float>> motion_results;


    int count=0;
    cv::Mat frame1;
    cv::Mat frame2;

    int frame_seq = 0; //check this reference --> what is the frame seq of results1 ???

    cap >> frame1; //check this reference --> Yide: seems to be inefficient, get the first and last frame

    double start_frame_ts = cap.get(cv::CAP_PROP_POS_MSEC); 
    
    //check this reference ---> how to get the best 2 frames ?? Additional function ??


    for(int i=0;i<fps;i++){
      cap>>frame2;

      count++;

    }

    if(frame2.empty()){
      break;
    }

    results1 = process_4(interpreter,frame1);
    output_id1=output_id(frame1, results1);
    generateIds(&results1, &id_list, &myList);
    update_lastseen(&results1);
    results2 = process_4(interpreter,frame2);
    output_id2=output_id(frame2, results2);
    start = std::chrono::system_clock::now();
    compare(
      frame1, &results1,output_id1,
      frame2, &results2,output_id2, &addition, &id_list, myList
    );
    
    moved_list=motion_detection_pair(results1, results2, 72, 0.80);
    for(int i=0;i<moved_list.size();i++){
      auto it = std::find(id_list.begin(), id_list.end(), moved_list[i]);
      if (it != id_list.end()) {
          // Element found
          //do nothing
      } else {
          // Element not found
          id_list.push_back(moved_list[i]);
      }
    }

    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    printf("The time for compare was s: %.10f\n", elapsed_seconds.count());

    output_id2=output_id(frame2, results2);
    std::string outputFileName = "./out/output" + std::to_string(count) + ".jpg";
    plotBboxes(frame2, results2, coco_names, outputFileName,id_list);
    update_lastseen(&results2);
    



    string output = "";
    last_frame=frame2;


    string meta_result1 = "";
    bool is_motion1 = false;
    
    

    if(id_list.size()>0){
    
      motion_results2=results2motion(&results2,&id_list);

      meta_result1 = get_meta_data(frame2, motion_results2); 
      std::cout << meta_result1 << std::endl;

      is_motion1 = true;
    }



    id_list.clear();

    while(true){ //check this reference --> this should only be two comparisons ??


      cap >> frame1;
      if (frame1.empty()) {
        break; 
      }

      double frame_ts = cap.get(cv::CAP_PROP_POS_MSEC);
      std::time_t frametime = static_cast<std::time_t>(frame_ts / 1000.0);
      std::tm localtm;
      localtime_r(&frametime, &localtm);

      // for(int i=0;i<fps;i++){
      //   if(i==fps/2){cap>>frame2;}
      //   cap>>end_frame;
      //   if(end_frame.empty()){
      //     break;
      //   }
      //   count++;

      // }


      for(int i=0;i<fps;i++){
        cap>>frame2;

        count++;

      }

      if(frame2.empty()){
        break;
      }

      //last_frame, reuslts2, output_ids2
      //do check with results2(previous frame) and results3
      std::vector<std::vector<float>> results3 = process_4(interpreter,frame1);
      std::vector<std::vector<float>> output_id3=output_id(frame1, results3);
      compare(
        last_frame, &results2,output_id2,
        frame1, &results3,output_id3, &addition,&id_list, myList
      );


      moved_list=motion_detection_pair(results2, results3, 40, 0.70);
      for(int i=0;i<moved_list.size();i++){
        auto it = std::find(id_list.begin(), id_list.end(), moved_list[i]);
        if (it != id_list.end()) {
            // Element found
            //do nothing
        } else {
            // Element not found
            id_list.push_back(moved_list[i]);
        }
      }


      output_id3=output_id(frame1, results3);
      update_lastseen(&results3);
      //now compare results3(frame1) and resutls4(frame2), then pass the results to resutls2
      std::vector<std::vector<float>> results4 = process_4(interpreter,frame2);
      std::vector<std::vector<float>> output_id4=output_id(frame2, results4);
      start = std::chrono::system_clock::now();
      compare(
        frame1, &results3,output_id3,
        frame2, &results4,output_id4, &addition,&id_list, myList
      );


      // moved_list=motion_detection_pair(results3, results4, 72, 0.80);
      moved_list=motion_detection_pair(results3, results4, 40, 0.70);
      for(int i=0;i<moved_list.size();i++){
        auto it = std::find(id_list.begin(), id_list.end(), moved_list[i]);
        if (it != id_list.end()) {
            // Element found
            //do nothing
        } else {
            // Element not found
            id_list.push_back(moved_list[i]);
        }
      }


      end = std::chrono::system_clock::now();
      elapsed_seconds = end - start;
      printf("The time for compare was s: %.10f\n", elapsed_seconds.count());
      output_id4=output_id(frame2, results4);
      std::string outputFileName = "./out/output" + std::to_string(count) + ".jpg";



      plotBboxes(frame2, results4, coco_names, outputFileName,id_list);
      update_lastseen(&results4);

      last_frame=frame2;
      results2=results4;
      output_id2=output_id4;



      //process metadata here
      string meta_result = "";
      bool is_motion = false;
      if(id_list.size()>0){
        std::cout << "get here 0 " <<std:: endl;
        motion_results=results2motion(&results4,&id_list);
        meta_result = get_meta_data(last_frame, motion_results); 
        std::cout << meta_result << std::endl;

        is_motion = true;
      }



      if(is_motion){
        
          //   //#ifdef VIPER_DEBUG_ENABLED
          //   //  std::cout << " FOUND MOTION PUSHING SEQ INTO MOTION QUEUE OF SIZE " << to_string(std_ml_image_event_deque.size()) << endl;
          //   //#endif
          //   //std::tm localtm = latest_frame->localtime;
          //   string localtm_str = datetm_to_str(localtm);
          //   //lock_guard<mutex> image_guard(std_ml_image_event_mutex); //update this reference uncomment it
            
          //   //meta_index is frameseq --> 
          //   //viper_sequence vip_seq = viper_sequence(segment_id,to_string(frameseq),frameseq, localtm_str); //update this reference uncomment it
          //   //if((std_ml_image_event_deque.size() > 0 && std_ml_image_event_deque.back().seq != segment_id) || (std_ml_image_event_deque.size() == 0)){ //don't push the event if it already is there for the current segment
          //   //std_ml_image_event_deque.push_back(vip_seq); //update this reference uncomment it
          //   //std_ml_image_event_index_map[segment_id] = std_ml_image_event_deque.size() - 1 + std_ml_image_event_deque_offset; //update this reference uncomment it
          //  // }

          //   string image_name = /*std_videos_save_dir +*/ to_string(segment_id) + "-" + to_string(frame_seq) + std_video_image_extension;
          //   string image_ref = "<rt7:Jpeg>" + image_name + "</rt7:Jpeg>";
          //   cv::imwrite(std_videos_save_dir +image_name, frame1, {cv::IMWRITE_JPEG_QUALITY,std_videos_jpeg_quality});
          //   output += "<rt7:Frame>"
          //   "<rt7:FrameSeq>" + to_string(frame_seq) + "</rt7:FrameSeq>" +
          //   "<rt7:FrameTime>" + to_string(int( (frame_ts - start_frame_ts) *1000000))+"</rt7:FrameTime>"
          //   + meta_result + image_ref + 
          //   "</rt7:Frame>";
        }

        remove_unfind(&results2);

        std::cout << "After removing elements:" << std::endl;
        for (const auto& innerVec : results2) {
            for (float value : innerVec) {
                std::cout << value << " ";
            }
            std::cout << std::endl;
        }



        id_list.clear(); // clean 
      
    }//end of video while loop

    // string std_ml_zone_def_str  = ""; //update this reference  //delete this it is a global var
    // string zonedef = "<rt7:ZoneDefinition>[" + std_ml_zone_def_str + "]</rt7:ZoneDefinition>"; //update this reference  //change this
    // string curr_segment_str = zonedef + output;

    // string meta_file = std_videos_save_dir + to_string(segment_id) + std_video_metadata_extension;
    // std::ofstream out(meta_file, std::ofstream::app);
    // out << curr_segment_str;
    // out.close();

    cap.release();


    break;


  }//end of infinite while loop


  return 0;

}




int main(int argc, char **argv){

  ml_processing_thread();
}






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
//   //motion object: 0,1,2,3,5,7
//   std::vector<float> myList={0.0, 1.0, 2.0, 3.0, 5.0, 7.0};
//   // create model
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
//   cv::VideoCapture cap("./4s.mp4"); 
//   if (!cap.isOpened()) {
//       std::cerr << "Error: Couldn't open input video!" << std::endl;
//       return -1;
//   }
//   // int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
//   // int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
//   int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
//   float addition=1;
//   float drop_threshold=20;
//   std::chrono::time_point<std::chrono::system_clock> beg, start, end, done, nmsdone;
//   std::chrono::duration<double> elapsed_seconds;
//   std::vector<std::vector<float>> results1;
//   std::vector<std::vector<float>> output_id1;
//   std::vector<std::vector<float>> results2;
//   std::vector<std::vector<float>> output_id2;
//   std::vector<int> id_list;
//   std::vector<int> moved_list;
//   cv::Mat last_frame;
//   int count=0;
//   cv::Mat frame1;
//   cv::Mat frame2;

//   cap >> frame1;
//   for(int i=0;i<fps;i++){
//     cap>>frame2;
//     count++;
//     if (frame2.empty()) {
//       break; 
//     }
//   }
//   results1 = process_4(interpreter,frame1);
//   output_id1=output_id(frame1, results1);
//   generateIds(&results1);
//   update_lastseen(&results1);
//   results2 = process_4(interpreter,frame2);
//   output_id2=output_id(frame2, results2);
//   compare(
//     frame1, &results1,output_id1,
//     frame2, &results2,output_id2, &addition, &id_list, myList
//   );
  
//   moved_list=motion_detection_pair(results1, results2, 72, 0.80);
//   for(int i=0;i<moved_list.size();i++){
//     auto it = std::find(id_list.begin(), id_list.end(), moved_list[i]);
//     if (it != id_list.end()) {
//         // Element found
//         //do nothing
//     } else {
//         // Element not found
//         id_list.push_back(moved_list[i]);
//     }
//   }

//   end = std::chrono::system_clock::now();
//   output_id2=output_id(frame2, results2);
//   std::string outputFileName = "./out/output" + std::to_string(count) + ".jpg";
//   plotBboxes(frame2, results2, coco_names, outputFileName,id_list);
//   update_lastseen(&results2);
//   id_list.clear();


//   last_frame=frame2;
//   while(true){
//     cap >> frame1;
//     for(int i=0;i<fps;i++){
//       cap>>frame2;
//       count++;
//       if (frame2.empty()) {
//         break; 
//       }
//     }
//     if (frame1.empty()) {
//       break; 
//     }
//     if (frame2.empty()) {
//       break; 
//     }
//     //last_frame, reuslts2, output_ids2
//     //do check with results2(previous frame) and results3
//     std::vector<std::vector<float>> results3 = process_4(interpreter,frame1);
//     std::vector<std::vector<float>> output_id3=output_id(frame1, results3);
//     compare(
//       last_frame, &results2,output_id2,
//       frame1, &results3,output_id3, &addition,&id_list, myList
//     );


//     moved_list=motion_detection_pair(results2, results3, 72, 0.80);
//     for(int i=0;i<moved_list.size();i++){
//       auto it = std::find(id_list.begin(), id_list.end(), moved_list[i]);
//       if (it != id_list.end()) {
//           // Element found
//           //do nothing
//       } else {
//           // Element not found
//           id_list.push_back(moved_list[i]);
//       }
//     }


//     output_id3=output_id(frame1, results3);
//     update_lastseen(&results3);
//     //now compare results3(frame1) and resutls4(frame2), then pass the results to resutls2
//     std::vector<std::vector<float>> results4 = process_4(interpreter,frame2);
//     std::vector<std::vector<float>> output_id4=output_id(frame2, results4);
//     start = std::chrono::system_clock::now();
//     compare(
//       frame1, &results3,output_id3,
//       frame2, &results4,output_id4, &addition,&id_list, myList
//     );


//     moved_list=motion_detection_pair(results3, results4, 72, 0.80);
//     for(int i=0;i<moved_list.size();i++){
//       auto it = std::find(id_list.begin(), id_list.end(), moved_list[i]);
//       if (it != id_list.end()) {
//           // Element found
//           //do nothing
//       } else {
//           // Element not found
//           id_list.push_back(moved_list[i]);
//       }
//     }


//     end = std::chrono::system_clock::now();
//     elapsed_seconds = end - start;
//     printf("The time for compare was s: %.10f\n", elapsed_seconds.count());
//     output_id4=output_id(frame2, results4);
//     std::string outputFileName = "./out/output" + std::to_string(count) + ".jpg";



//     plotBboxes(frame2, results4, coco_names, outputFileName,id_list);
//     update_lastseen(&results4);
//     id_list.clear();
//     last_frame=frame2;
//     results2=results4;
//     output_id2=output_id4;

//   }
//   cap.release();
//   return 0;

// }



