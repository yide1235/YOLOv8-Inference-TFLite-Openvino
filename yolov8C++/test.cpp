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


typedef cv::Point3_<float> Pixel;


auto mat_process2(cv::Mat src, uint width, uint height) -> cv::Mat
{
  // convert to float; BGR -> RGB
  cv::Mat dst2;
  cout << "Creating dst" << endl;
  // src.convertTo(dst, CV_32FC3);
  cout << "Creating dst2" << endl;
  cv::cvtColor(src, dst2, cv::COLOR_BGR2RGB);
  cout << "Creating dst3" << endl;


  cv::Mat normalizedImage(dst2.rows, dst2.cols, CV_32FC3);
  // cv::Mat normalizedImage(dst2.rows, dst2.cols, CV_16FC3);
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
    // cv::Size new_shape(640, 640);
    cv::Size new_shape(10,10);

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



void normalize(Pixel &pixel)
{
  pixel.x = (pixel.x / 255.0);
  pixel.y = (pixel.y / 255.0);
  pixel.z = (pixel.z / 255.0);
}

auto mat_process(cv::Mat src, uint width, uint height) -> cv::Mat
{
  // convert to float; BGR -> RGB
  cv::Mat dst;
  cout << "Creating dst" << endl;
  src.convertTo(dst, CV_32FC3);
  cout << "Creating dst2" << endl;
  cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);
  cout << "Creating dst3" << endl;

  // normalize to -1 & 1
  Pixel *pixel = dst.ptr<Pixel>(0, 0);
  cout << "Creating dst4" << endl;
  const Pixel *endPixel = pixel + dst.cols * dst.rows;
  cout << "Creating dst5" << endl;
  for (; pixel != endPixel; pixel++)
    normalize(*pixel);

  // resize image as model input
  cout << "Creating dst6" << endl;
  cv::resize(dst, dst, cv::Size(width, height));
  cout << "Creating dst7" << endl;
  return dst;
}



string process_frame(const char *img, int height, int width)
{

  unsigned char *output_pixels;
  cout << "Creating raw data" << endl;
  Mat raw_data(1, sizeof(img), CV_8SC1, (void *)img);
  cout << "Creating Input img" << endl;

  cv::Mat img_tst = cv::imread("image0frame0.jpg");
  cv::Mat input_img = mat_process(img_tst, 320, 320);
  cout << "Creating img decode" << endl;
  Mat decode = imdecode(raw_data, IMREAD_ANYDEPTH);

  cout << "Decoded the cv matrix " << endl;
  // stbir_resize_uint8(latest_frame->img,1920,1080,8,output_pixels,320,320,8,3);

  // int ret = stbir_resize_uint8(latest_frame->img,1920,1080,8,output_pixels,320,320,8,3);

  int inp_image_width = width;
  int inp_image_height = height;
  const char *model_path = "yolov5.tflite";
  // loads the model
  std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path);
  cout << "Got model path " << endl;
  int channels = 3;

  // builds the interpreter to run the model (all should be made with InterpreterBuilder)
  tflite::ops::builtin::BuiltinOpResolver resolver;

  tflite::InterpreterBuilder builder(*model, resolver);

  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  if (!interpreter)
  {
    cout << "Failed to construct interpreter\n";
    exit(-1);
  }

  cout << "Built interpreter!" << endl;

  interpreter->SetAllowFp16PrecisionForFp32(false);

  cout << "Loaded model!" << endl;
  interpreter->ResizeInputTensor(0, {1, inp_image_width, inp_image_height, 3});
  if (interpreter->AllocateTensors() != kTfLiteOk) // must be called at initialization or after the input tensors are reassigned
  {
    cout << "Failed to allocate tensors!\n";
    exit(-1);
  }
  // cv::Mat test_img = cv::imread("models/out_out_download.jpg");
  tflite::PrintInterpreterState(interpreter.get());

  cout << "Allocated tensors and displayed interpreter state!" << endl;

  float *input_tensor = interpreter->typed_input_tensor<float>(0);

  cout << "Allocated tensors and displayed interpreter state!" << endl;

  //    double scale_w = (double)inp_image_width / latest_frame.rows;
  //     double scale_h = (double)inp_image_height / latest_frame.cols;

  //     double scale;
  //    if (scale_w < scale_h)
  //     {
  //         scale = scale_w;
  //     }
  //     else
  //     {
  //         scale = scale_h;
  //     }

  //   cv::resize(latest_frame,  latest_frame, cv::Size(0, 0), scale, scale);
  //     cv::Mat test_img_r(inp_image_width , inp_image_height,  latest_frame.type());
  //     int top_bottom_fill = (inp_image_height -  latest_frame.rows) / 2;
  //     int left_right_fill = (inp_image_width -  latest_frame.cols) / 2;
  //     cv::copyMakeBorder( latest_frame, test_img_r, top_bottom_fill, top_bottom_fill, left_right_fill, left_right_fill, CV_HAL_BORDER_CONSTANT, 0);

  //     int img_size = test_img_r.rows * test_img_r.cols * 3;

  float *test_img_arr = (float *)malloc(inp_image_height * sizeof(float));

  for (int i = 0; i < inp_image_height; i++)
  {
    test_img_arr[i] = (float)img[i];
  }

  memcpy(input_tensor, test_img_arr, inp_image_height * sizeof(float));

  //     auto t1 = chrono::high_resolution_clock::now();

  interpreter->Invoke();
  int output = interpreter->outputs()[0];
  TfLiteIntArray *output_dims = interpreter->tensor(output)->dims;
  // // assume output dims to be something like (1, 1, ... ,size)
  auto output_size = output_dims->data[output_dims->size - 1];

  //   float *input_tensor1 = interpreter->typed_input_tensor<float>(0);

  cout << "Number of inputs: " << interpreter->inputs().size() << endl;
  float *output2 = interpreter->typed_output_tensor<float>(0);

  printf("Result is: %f\n", *output2);
  free(test_img_arr); // delete the array

  return "";
}

// template <typename T>
// auto cvtTensor(TfLiteTensor *tensor) -> vector<T>;

// auto cvtTensor(TfLiteTensor *tensor) -> vector<float>
// {
//   int nelem = 1;
//   for (int i = 0; i < tensor->dims->size; ++i)
//   {
//     cout << "DIM IS " << tensor->dims->data[i] << endl;
//     nelem *= tensor->dims->data[i];
//   }

//   cout << "NELEM IS " << nelem << endl;
//   vector<float> data(tensor->data.f, tensor->data.f + nelem);
//   cout << "FLATTENED VALS " << endl;
//   return data;
// }



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


// template<typename T>
// auto cvtTensor(TfLiteTensor* tensor) -> vector<T>;

// auto cvtTensor(TfLiteTensor* tensor) -> vector<float>{
//     int nelem = 1;
//     for(int i=0; i<tensor->dims->size; ++i)
//         nelem *= tensor->dims->data[i];
//     vector<float> data(tensor->data.f, tensor->data.f+nelem);
//     return data;
// }



cv::Mat process_4(string outfile, string image, vector<Rect> &nmsrec, vector<int> &pick, vector<int> &ids, cv::Mat *inp = NULL)
{


    // create model
  std::unique_ptr<tflite::FlatBufferModel> model =
  tflite::FlatBufferModel::BuildFromFile(("./yolov8n_integer_quant.tflite"));

  //comment this if not use npu
  // auto ext_delegate_option = TfLiteExternalDelegateOptionsDefault("/usr/lib/libvx_delegate.so");
  // auto ext_delegate_ptr = TfLiteExternalDelegateCreate(&ext_delegate_option);


  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);

  interpreter->SetAllowFp16PrecisionForFp32(true);

  //comment this if not use npu
  // interpreter->ModifyGraphWithDelegate(ext_delegate_ptr);

  interpreter->AllocateTensors();

  //setupInput(interpreter);

  cout << " Got model " << endl;
  // get input & output layer
  TfLiteTensor *input_tensor = interpreter->tensor(interpreter->inputs()[0]);
  cout << " Got input " << endl;
  TfLiteTensor *output_box = interpreter->tensor(interpreter->outputs()[0]);
  cout << " Got output " << endl;
  TfLiteTensor *output_score = interpreter->tensor(interpreter->outputs()[1]);
  cout << " Got output score " << endl;

  const uint HEIGHT = input_tensor->dims->data[1];
  const uint WIDTH = input_tensor->dims->data[2];
  const uint CHANNEL = input_tensor->dims->data[3];
  cout << "H " << HEIGHT << " W " << WIDTH << " C " << CHANNEL << endl;

  // read image file
  std::chrono::time_point<std::chrono::system_clock> beg, start, end, done, nmsdone;
  std::chrono::duration<double> elapsed_seconds;
  start = std::chrono::system_clock::now();

  cv::Mat img;
  if (inp == NULL)
  {
    cout << "Getting image from file " << endl;
    img = cv::imread(image);
  }
  else
  {
    cout << "Getting image from input " << endl;
    img = *inp;
  }

  end = std::chrono::system_clock::now();
  elapsed_seconds = end - start;
  printf("Read matrix from file s: %.10f\n", elapsed_seconds.count());

  start = std::chrono::system_clock::now();

  //original image preprocessing
  // cv::Mat inputImg = mat_process(img, WIDTH, HEIGHT); // could be input is modified by this function
  
  //python way image preprocessing

  cv::Mat inputImg = letterbox(img, WIDTH, HEIGHT);

  inputImg = mat_process(img, WIDTH, HEIGHT);
  

  cout << " Got image " << endl;

  end = std::chrono::system_clock::now();
  elapsed_seconds = end - start;
  printf("Process Matrix to RGB s: %.10f\n", elapsed_seconds.count());

  start = std::chrono::system_clock::now();
  // cout << " GOT INPUT IMAGE " << endl;

  // flatten rgb image to input layer.
  float *inputImg_ptr = inputImg.ptr<float>(0);
  memcpy(input_tensor->data.f, inputImg.ptr<float>(0),
         WIDTH * HEIGHT * CHANNEL * sizeof(float));

  cout << " GOT MEMCPY " << endl;
  // compute model instance

  interpreter->Invoke();
  end = std::chrono::system_clock::now();
  elapsed_seconds = end - start;
  printf("invoke interpreter s: %.10f\n", elapsed_seconds.count());

  float *output1 = interpreter->typed_output_tensor<float>(0);
  // float* output2 = interpreter->typed_output_tensor<float>(1);

  vector<cv::Rect> rects;
  vector<vector<float>> recvec;
  vector<float> scores;
  // vector<int> ids;
  vector<int> nms;
  map<int, vector<vector<float>>> map_rects;
  map<int, vector<float>> map_scores;

  float max_class_score = 0;
  int max_class = -1;
  float MAX_ACCEPT_SCORE = 0.40;
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
  cout << " bounding boxes " << endl;
  printf("%f ", output1[0]);
  printf("%f ", output1[1]);
  printf("%f ", output1[2]);
  printf("%f ", output1[3]);
  printf("%f ", output1[4]);
  printf("%f ", output1[80]);
  printf("%f ", output1[81]);
  printf("%f ", output1[82]);
  printf("%f ", output1[83]);

  // printf("%f ", output1[8400 * 1]);
  // printf("%f ", output1[8400 * 2]);
  // printf("%f ", output1[8400 * 3]);
  // printf("%f ", output1[8400 * 4]);
  // printf("%f ", output1[8400 * 80]); //0
  // printf("%f ", output1[8400 * 81]); //0
  // printf("%f ", output1[8400 * 82]); //0
  // printf("%f ", output1[8400 * 83]); //0
  cout << " done print outputs " << endl;

  cout << " dim1 is " << dim1 << endl;

  return img;

}


string motion_detect(cv::Mat mat1, cv::Mat mat2, vector<Rect> nmsrec, vector<int> pick, vector<int> ids, vector<int> &motion)
{

  std::chrono::time_point<std::chrono::system_clock> beg, start, end, done, nmsdone;
  std::chrono::duration<double> elapsed_seconds;
  start = std::chrono::system_clock::now();

  cv::Mat dst, cln;
  cout << " Starting abs diff " << endl;
  absdiff(mat1, mat2, dst);
  end = std::chrono::system_clock::now();
  elapsed_seconds = start - end;
  printf("The time for abs diff was s: %.10f\n", elapsed_seconds.count());

  cln = dst.clone();
  imwrite("./diff_outputbeg0.jpg", cln);

  start = std::chrono::system_clock::now();
  // cout << " Got image diff " << endl;

  vector<double> eval;
  cv::Scalar eval_mean, eval_stdev;
  cv::Scalar mat_mean, mat_stdev;
  int pick_index = 0;
  double cum_area = 0;
  double total_mean = 0;
  double total_stdev = 0;
  int channels = dst.channels();
  // cout << " Iterating over rec with channels " << channels << endl;

  // mat_mean = cv::sum(dst);
  // meanStdDev(dst,eval_mean,eval_stdev);
  // cout << " THE MAT_MEAN WAS................................." << mat_mean[0] << endl;

  for (cv::Rect rect : nmsrec)
  {
    float x = rect.x;           // rect.x/1920.0*320.0;
    float y = rect.y;           // rect.y/1080.0*320.0;
    float width = rect.width;   // rect.width/1920.0*320.0;
    float height = rect.height; // rect.height/1080.0*320.0;
    // cout << "x is " << x << " Y is " << y << " Width is " << width << " Height is " << height << endl;

    // adjust the coordinates to fit into the image
    if (x < 0)
    {
      x = 0;
    }

    if (y < 0)
    {
      y = 0;
    }

    if (x + width > dst.cols)
    {
      width = dst.cols - x - 1;
      // cout << " Width is now " << width << endl;
    }
    if (y + height > dst.rows)
    {
      height = dst.rows - y - 1;
      // cout << " Height is now " << height << endl;
    }
    // end adjustments

    cv::Rect mod_rect = Rect(x, y, width, height);
    // cout << " Curr rect is x,y,width,height " << rect.x << " " << rect.y << " " << " " << rect.width << " " << rect.height << endl;
    // cout << " Curr mat is cols, rows" << dst.cols << " , " << dst.rows << endl;
    Mat roi = dst(mod_rect);
    cv::meanStdDev(roi, eval_mean, eval_stdev);
    // eval_mean = cv::sum(roi);
    double sum = 0;
    double sample_stdev = 0;
    for (int i = 0; i < channels; i++)
    {
      sum += eval_mean[i];
      // cout << "Eval Mean i " << eval_mean[i] << " pick index " << pick_index << endl;
    }

    /*for(int i=0; i<channels; i++){
      sample_stdev += ((eval_mean[i] - sum/channels) * (eval_mean[i] - sum/channels))/channels;
      cout << "sample stddev cumulative " << sample_stdev << " pick index " << pick_index << endl;
    }

    sample_stdev = sample_stdev / (mod_rect.width * mod_rect.height);
    sample_stdev = sqrt(sample_stdev);
    cout << " total std dev " << sample_stdev << " for rect " << pick_index << endl;*/

    eval.push_back(sum); // divide the sum of means by the number of channels to get the channel mean

    roi.setTo(0); // zero out the matrix
    pick_index++;
  }

  cv::meanStdDev(dst, mat_mean, mat_stdev);
  // mat_mean = cv::sum(dst);
  // cout << " THE MAT_MEAN2 WAS..............................." << mat_mean[0] << endl;
  for (int i = 0; i < channels; i++)
  {
    total_mean += mat_mean[i];
    total_stdev += mat_stdev[i];
    // cout << " Total mat mean " << mat_mean[i] << endl;
    // cout << " Total mean " << total_mean << " " << total_stdev << endl;
  }

  /*for(int i=0; i<channels; i++){
      total_stdev  += ((mat_mean[i] - total_mean/channels) * (eval_mean[i] - total_mean/channels))/channels;
      cout << "sample stddev cumulative " << total_stdev  << " pick index " << pick_index << endl;
  }

  total_stdev = total_stdev / (dst.cols * dst.rows);
  cout << "sample stddev cumulative after col row division " << total_stdev  << endl;
  total_stdev = sqrt(total_stdev);*/
  double motion_cutoff = total_mean + 1.7 * total_stdev;
  // cout << " The motion cutoff is " << motion_cutoff << " total_stddev is " << total_stdev << endl;

  end = std::chrono::system_clock::now();
  elapsed_seconds += start - end;

  printf("The time for motion diff was s: %.10f\n", elapsed_seconds.count());

  pick_index = 0;
  for (cv::Rect rect : nmsrec)
  {
    cout << " Testing rect as pick index " << pick_index << " which has class id " << ids[pick[pick_index]] << " and motion val " << eval[pick_index] << endl;
    if (eval[pick_index] > motion_cutoff)
    {
      motion.push_back(pick_index);
      // cout << " Rect as pick index " << pick_index << " shows motion with class id " << ids[pick[pick_index]] << endl;
      cv::rectangle(cln, rect, cv::Scalar(0, 255, 0), 3);

      // to_string(ids[pick[pick_index]])
      cv::putText(cln, "ind" + to_string(pick_index), cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(118, 185, 0), 2);
    }
    pick_index++;
  }

  cout << " Computed abs diff " << endl;
  imwrite("./diff_output0.jpg", cln);
  cout << " Wrote diff image zack did this " << endl;
  /*
    opencv absdiff get max change
    compare the max frame
    check absdiff of objects in max frame
    determine motion
    develop motion metadata
  */

  return "";
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

/*cv::Mat get_mat(char* jpeg_img){
  cv::Mat rawData(1080, 1920, CV_8FC3, (void*)jpeg_img);
  //cv::Mat inputImg = imdecode(rawData, IMREAD_ANYDEPTH);
  cout << "Got matrix and converted it to 1920 1080 CV_8SC3..." << endl;
  return rawData;
}*/

int main(int argc, char **argv)
{


  std::cout << " Tensorflow Test " << endl;

  // int height =320;
  // int width = 320;
  // string file_name = "image0frame0.jpg";
  // const char* img = read_image(file_name);
  // cout << "Got image char array " << endl;
  // string resp = process_frame(img, height, width);
  // string imgf1 = "image0frame0.jpg";
  string imgf1 = "3333.jpg";
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

  auto start2 = chrono::high_resolution_clock::now();

  int filesize = 0;
  char *img1 = read_image(imgf1, filesize);
  cv::Mat img1_mat = convert_image(img1, 1080, 1920, filesize);
  char *img2 = read_image(imgf2, filesize);
  cv::Mat img2_mat = convert_image(img2, 1080, 1920, filesize);


//   cv::Mat pmat1 = process_2(interpreter, "","./result0.jpg", nmsrec1, pick1, ids1,  &img1_mat);

  // cv::Mat pmat1= detect_frame("", "./result0.jpg", nmsrec1, pick1, ids1, &img1_mat);
  cv::Mat pmat1= process_4("", "./result0.jpg", nmsrec1, pick1, ids1, &img1_mat);
  auto end2 = chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds1 = end2 - start2;
  printf("The time for yolov5 detected frame was s: %.10f\n", elapsed_seconds1.count());



  //cv::Mat pmat2 = process_4(interpreter,"", "./result1.jpg", nmsrec2, pick2, ids2,  &img2_mat);
  return 0;

  // // string test_file = "viper_snapshot.jpg";
  // // char* img_test = read_image(test_file);
  // // cv::Mat result = convert_image(img_test);
  // // cv::Mat pmat3 = process_2(test_file, "./result3.jpg", nmsrec3, pick3, ids3, &result);
  // // cout << "Processed the result " << endl;

  // // cv::Mat i1 = cv::imread(imgf1);
  // // cv::Mat i2= mat_process(i1, 1920, 1080); //could be input is modified by this function
  // // cv::Mat j1 = cv::imread(imgf2);
  // // cv::Mat j2= mat_process(j1, 1920, 1080); //could be input is modified by this function

  // // char* img1 = read_image(imgf1);
  // // char* img2 = read_image(imgf2);
  // std::cout << " Got images " << endl;
  // // cv::Mat mat1 = get_mat(img1);
  // // cv::Mat mat2 = get_mat(img2);
  // cout << " Starting motion detect " << endl;
  // motion_detect(img1_mat, img2_mat, nmsrec2, pick2, ids2, motion2);
  // cout << "Processing motion detection " << endl;

  // cout << " Deleting malloc image " << endl;
  // // delete img1;
  // // delete img2;
  // cout << " Done Delete " << endl;

  // auto t1 = chrono::high_resolution_clock::now();

  // // char * img1 = read_image(imgf1, filesize);
  // Mat imgv_mat = convert_image(img1, 1080, 1920, filesize);
  // cv::Mat out_t, outt1, outt2, outt3, outt4;
  // cv::Scalar mat_mean, mat_stdev;
  // absdiff(img1_mat, img2_mat, out_t);
  // double total_mean = 0;
  // double total_stdev = 0;
  // cv::meanStdDev(out_t, mat_mean, mat_stdev);
  // // mat_mean = cv::sum(dst);
  // // cout << " THE MAT_MEAN2 WAS..............................." << mat_mean[0] << endl;
  // for (int i = 0; i < 3; i++)
  // {
  //   total_mean += mat_mean[i];
  //   total_stdev += mat_stdev[i];
  //   cout << " Total mat mean " << mat_mean[i] << endl;
  //   cout << " Total mean " << total_mean << " " << total_stdev << endl;
  // }

  // threshold(out_t, outt1, 0, 255, 3);
  // auto t2 = chrono::high_resolution_clock::now();

  // cv::Mat avg = 0.9 * out_t + 0.1 * imgv_mat;
  // auto t3 = chrono::high_resolution_clock::now();

  // cv::imwrite("./outt1.jpg", outt1);

  // threshold(out_t, outt2, 50, 255, 3);
  // cv::imwrite("./outt2.jpg", outt2);

  // threshold(out_t, outt3, 75, 255, 3);
  // cv::imwrite("./outt3.jpg", outt3);

  // threshold(out_t, outt4, 200, 255, 3);
  // cv::imwrite("./outt4.jpg", outt4);

  // std::chrono::duration<double> elapsed_seconds = t2 - t1;
  // std::chrono::duration<double> elapsed_seconds2 = t3 - t2;

  // printf("The time for motion diff was s: %.10f\n", elapsed_seconds.count());
  // printf("The time to compute average matrix was s: %.10f\n", elapsed_seconds2.count());
  
}