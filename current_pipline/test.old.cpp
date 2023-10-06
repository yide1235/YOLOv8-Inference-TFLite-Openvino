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

template <typename T>
auto cvtTensor(TfLiteTensor *tensor) -> vector<T>;

auto cvtTensor(TfLiteTensor *tensor) -> vector<float>
{
  int nelem = 1;
  for (int i = 0; i < tensor->dims->size; ++i)
  {
    cout << "DIM IS " << tensor->dims->data[i] << endl;
    nelem *= tensor->dims->data[i];
  }

  cout << "NELEM IS " << nelem << endl;
  vector<float> data(tensor->data.f, tensor->data.f + nelem);
  cout << "FLATTENED VALS " << endl;
  return data;
}

cv::Mat process_2(string infile, string outfile, vector<Rect> &nmsrec, vector<int> &pick, vector<int> &ids, cv::Mat *inp = NULL)
{
  // create model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile("yolov5.tflite");
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);

  // TEST ADDING BeLOW EVAN
  tflite::StatefulNnApiDelegate::Options options;
  
  auto delegate = tflite::evaluation::CreateNNAPIDelegate(options);
  // auto delegate = tflite::evaluation::CreateNNAPIDelegate(options);
  if (!delegate)
  {
    cout << endl
         << endl
         << "ERROR IN MAKING THE DELEGATE FOR NNAPI!!!" << endl
         << endl;
    exit(-1);
  }
  else
  {
    interpreter->ModifyGraphWithDelegate(std::move(delegate));
    cout << endl
         << endl
         << "SUCCESSFULLY ADDED NNAPI DELEGATE TO INTERPRETER..." << endl
         << endl;
  }
  //  END OF TEST ADDING

  interpreter->AllocateTensors();

  // get input & output layer
  TfLiteTensor *input_tensor = interpreter->tensor(interpreter->inputs()[0]);
  TfLiteTensor *output_box = interpreter->tensor(interpreter->outputs()[0]);
  TfLiteTensor *output_score = interpreter->tensor(interpreter->outputs()[1]);

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

  end = std::chrono::system_clock::now();
  elapsed_seconds = end - start;
  printf("Read matrix from file s: %.10f\n", elapsed_seconds.count());

  start = std::chrono::system_clock::now();

  cv::Mat inputImg = mat_process(img, WIDTH, HEIGHT); // could be input is modified by this function

  end = std::chrono::system_clock::now();
  elapsed_seconds = end - start;
  printf("Process Matrix to RGB s: %.10f\n", elapsed_seconds.count());

  start = std::chrono::system_clock::now();
  // cout << " GOT INPUT IMAGE " << endl;

  // flatten rgb image to input layer.
  float *inputImg_ptr = inputImg.ptr<float>(0);
  memcpy(input_tensor->data.f, inputImg.ptr<float>(0),
         WIDTH * HEIGHT * CHANNEL * sizeof(float));

  // cout << " GOT MEMCPY " << endl;
  //  compute model instance

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
  float MAX_ACCEPT_SCORE = 0.20;
  float bbox_score = 0;

  int nelem = 1;
  for (int i = 0; i < output_box->dims->size; ++i)
  {
    // cout << "DIM IS " << output_box->dims->data[i] << endl;
    nelem *= output_box->dims->data[i];
  }

  for (int oi = 0; oi < nelem; oi++)
  {
    // printf("%f ", output1[oi]);
    if (oi % 85 == 0)
    {
      // printf("\n");
      if (max_class_score > MAX_ACCEPT_SCORE)
      {
        // printf("\n");
        // for(int oj=oi-85; oj<oi; oj++){
        //   printf("%f ", output1[oj]);
        // }
        // printf("\n");
        const float cx = output1[oi - 85];
        const float cy = output1[oi - 84];
        const float w = output1[oi - 83];
        const float h = output1[oi - 82];
        // cout << " xywh HEIGHT WIDTH COLS ROWS " << cx << " " << cy << " " << w << " " << h << " " << HEIGHT << " " << WIDTH << " " << img.cols << " " << img.rows << endl;
        const float xmin = ((cx - (w / 2))) * img.cols;
        const float ymin = ((cy - (h / 2))) * img.rows;
        const float xmax = ((cx + (w / 2))) * img.cols;
        const float ymax = ((cy + (h / 2))) * img.rows;
        rects.emplace_back(cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin));
        // cout << xmin<< " " << ymin << " " << xmax-xmin << " " << ymax-ymin << endl;
        vector<float> currvec = {xmin, ymin, xmax, ymax};
        recvec.emplace_back(currvec);
        // rects.emplace_back(cv::Rect((output1[oi-85]-(0.5*output1[oi-83]))*img.cols, (output1[oi-84]-(0.5*output1[oi-82]))*img.rows, output1[oi-83]*img.cols, output1[oi-82]*img.rows));
        scores.emplace_back(max_class_score);
        // ids.emplace_back(max_class);
        // cout << (output1[oi-85]-(0.5*output1[oi-83]))*img.cols << " " << (output1[oi-84]-(0.5*output1[oi-82]))*img.rows << " " << output1[oi-83]*img.cols << " " << output1[oi-82]*img.rows << " " << max_class << " " << max_class_score << endl;
        // cout << output1[oi-85] << " " << output1[oi-84] << " " << output1[oi-83] << " " << output1[oi-82] << " " << endl;
        // cout << " MAX CLASS  MAX SCORE " << max_class << " " << max_class_score << endl;

        // cout << " BBOX SCORE IS " << output1[oi-86] << endl;
        // cout << " FIRST CLASS VALUE IS " << output1[oi-85] << endl;

        if (map_rects.find(max_class) == map_rects.end())
        {
          map_rects[max_class] = vector<vector<float>>();
          map_rects[max_class].emplace_back(currvec);
          map_scores[max_class] = vector<float>();
          map_scores[max_class].emplace_back(max_class_score);
          // not found
        }
        else
        {
          // found
          map_rects[max_class].emplace_back(currvec);
          map_scores[max_class].emplace_back(max_class_score);
        }
      }

      max_class_score = 0;
      max_class = -1;
      bbox_score = output1[oi + 4];
      oi += 5;
    }

    float test_score = output1[oi] * bbox_score;
    if (test_score > max_class_score)
    {
      max_class_score = test_score;
      max_class = (oi % 85) - 5;
    }
  }

  done = std::chrono::system_clock::now();
  elapsed_seconds = done - end;
  printf("Create Boxes from interpreter outputs s: %.10f\n", elapsed_seconds.count());

  // cv::dnn::NMSBoxes(rects,scores,0.2,0.4,nms);
  // vector<int> pick;
  /*vector<Rect>*/
  int prev_count = 0;
  for (auto element : map_rects)
  {
    // int class_id = element.first;
    // nmsrec.emplace_back(nms_vec(element.second,0.5, pick));
    vector<Rect> temp_nms_rec;
    vector<int> temp_pick;
    cout << "1 Size of temp nms rec is " << temp_nms_rec.size() << endl;
    cout << "1 Size of temp pick is " << temp_pick.size() << endl;
    cout << "1 Size of nms rec is " << nmsrec.size() << endl;
    cout << "1 Size of vec rect is " << element.second.size() << endl;
    temp_nms_rec = nms_vec(element.second, 0.5, temp_pick);

    copy(temp_nms_rec.begin(), temp_nms_rec.end(), back_inserter(nmsrec));
    cout << "2 Size of temp nms rec is " << temp_nms_rec.size() << endl;
    cout << "2 Size of temp pick is " << temp_pick.size() << endl;
    cout << "2 Size of nms rec is " << nmsrec.size() << endl;
    cout << "2 Prev count is " << prev_count << endl;
    for (int index = 0; index < temp_pick.size(); index++)
    {
      // cout << " Temp pick value is " << temp_pick[index] << endl;
      cout << " Emplace back with value " << index + prev_count << endl;
      pick.emplace_back(index + prev_count);
    }
    // copy(temp_pick.begin(), temp_pick.end(), back_inserter(pick));
    ids.insert(ids.end(), temp_nms_rec.size(), element.first);
    prev_count += temp_nms_rec.size();
    cout << " Size of Ids is " << ids.size() << endl;
    // ids.emplace_back()
    // nmsrec.emplace_back(temp_nms_rec);
    // pick.emplace_back(temp_pick);
  }

  // nmsrec = nms_vec(recvec,0.5, pick);

  nmsdone = std::chrono::system_clock::now();
  elapsed_seconds = nmsdone - done;
  printf("Process nms s: %.10f\n", elapsed_seconds.count());

  cv::Mat clnimg = img.clone();
  int pick_index = 0;
  for (cv::Rect rect : nmsrec)
  {
    cv::rectangle(clnimg, rect, cv::Scalar(0, 255, 0), 3);
    cv::putText(clnimg, to_string(ids[pick[pick_index]]), cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(118, 185, 0), 2);
    cout << " New rect is " << rect << endl;
    cout << " New index is " << pick[pick_index] << endl;
    cout << " Class is " << ids[pick[pick_index]] << endl;
    pick_index++;
  }

  cv::imwrite(outfile, clnimg);
  // cout << " WROTE IMAGE " << endl;

  // printf("Result is: %f\n", *output2);

  // std::cout << "OUTPUT BOX DIMS ARE " << output_box->dims->size << "\n"; // Tensor<type: float shape: [] values: 30>
  // std::cout << "OUTPUT BOX DIMS ARE " << output_box->dims->data << "\n"; // Tensor<type: float shape: [] values: 30>
  // std::cout << "OUTPUT SCORE DIMS ARE " << output_score->dims << "\n"; // Tensor<type: float shape: [] values: 30>

  /*
  vector<float> box_vec = cvtTensor(output_box);
  //vector<float> score_vec = cvtTensor(output_score);
  cout << " GOT CVT TENSOR " << endl;

  vector<size_t> result_id;
  auto it = std::find_if(std::begin(score_vec), std::end(score_vec),
                         [](float i){return i > 0.6;});
  while (it != std::end(score_vec)) {
      result_id.emplace_back(std::distance(std::begin(score_vec), it));
      it = std::find_if(std::next(it), std::end(score_vec),
                        [](float i){return i > 0.6;});
  }

  cout << " GOT RESULT ID " << endl;

  vector<cv::Rect> rects;
  vector<float> scores;
  for(size_t tmp:result_id){
      const int cx = box_vec[4*tmp];
      const int cy = box_vec[4*tmp+1];
      const int w = box_vec[4*tmp+2];
      const int h = box_vec[4*tmp+3];
      const int xmin = ((cx-(w/2.f))/WIDTH) * img.cols;
      const int ymin = ((cy-(h/2.f))/HEIGHT) * img.rows;
      const int xmax = ((cx+(w/2.f))/WIDTH) * img.cols;
      const int ymax = ((cy+(h/2.f))/HEIGHT) * img.rows;
      rects.emplace_back(cv::Rect(xmin, ymin, xmax-xmin, ymax-ymin));
      scores.emplace_back(score_vec[tmp]);
  }

  cout << " GOT RECTS " << endl;

  vector<int> ids;

  //cv::dnn::NMSBoxes(rects, scores, 0.6, 0.4, ids);
  //for(int tmp: ids)
  //   cv::rectangle(img, rects[tmp], cv::Scalar(0, 255, 0), 3);
  for(cv::Rect rect: rects){
      cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 3);
  }

  cout << " GOT CV RECT " << endl;
  cv::imwrite("./result.jpg", img);
  cout << " WROTE IMAGE " << endl;
  */

  return inputImg;
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

cv::Mat process_3(string infile, string outfile, vector<Rect> &nmsrec, vector<int> &pick, vector<int> &ids, cv::Mat *inp = NULL)
{
  // create model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile("yolov8n_int8.tflite");


  auto ext_delegate_option = TfLiteExternalDelegateOptionsDefault("libvx_delegate.so");


  auto ext_delegate_ptr = TfLiteExternalDelegateCreate(&ext_delegate_option);

  // Build the interpreter with the InterpreterBuilder.
  // Note: all Interpreters should be built with the InterpreterBuilder,
  // which allocates memory for the Interpreter and does various set up
  // tasks so that the Interpreter can read the provided model.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  //resolver.AddCustom(kNbgCustomOp, tflite::ops::custom::Register_VSI_NPU_PRECOMPILED());

  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  //TFLITE_EXAMPLE_CHECK(npu_interpreter != nullptr);
  interpreter->ModifyGraphWithDelegate(ext_delegate_ptr);

  // Allocate tensor buffers.
  //TFLITE_EXAMPLE_CHECK(npu_interpreter->AllocateTensors() == kTfLiteOk);
  //TFLITE_LOG(tflite::TFLITE_LOG_INFO, "=== Pre-invoke NPU Interpreter State ===");
  
  ////THIS IS THE LOG FILE
  tflite::PrintInterpreterState(interpreter.get());
  ///

  // Fill input buffers
  // TODO(user): Insert code to fill input tensors.
  // Note: The buffer of the input tensor with index `i` of type T can
  // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);`

  //setupInput(argc, argv, npu_interpreter,is_use_cache_mode);

  // Run inference
  //TFLITE_EXAMPLE_CHECK(npu_interpreter->Invoke() == kTfLiteOk);

  // Get performance
  // {
  //   const uint32_t loop_cout = 10;
  //   auto start = std::chrono::high_resolution_clock::now();
  //   for (uint32_t i = 0; i < loop_cout; i++) {
  //     npu_interpreter->Invoke();
  //   }
  //   auto end = std::chrono::high_resolution_clock::now();
  //   std::cout << "[NPU Performance] Run " << loop_cout << " times, average time: " << (end - start).count() << " ms" << std::endl;
  // }

  //TFLITE_LOG(tflite::TFLITE_LOG_INFO, "=== Post-invoke NPU Interpreter State ===");
  //tflite::PrintInterpreterState(npu_interpreter.get());


  //tflite::ops::builtin::BuiltinOpResolver resolver;
  //std::unique_ptr<tflite::Interpreter> interpreter;
  //tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);

  // TEST ADDING BeLOW EVAN
  /*
  cout<<"AT EVAN CODE..."<<endl;

  const char* delegate_so_path = "libvx_delegate.so";
  void* delegate_lib = dlopen(delegate_so_path, RTLD_LAZY);
  if (!delegate_lib) {
      std::cerr << "Failed to load the custom delegate: " << dlerror() << std::endl;
  }

  cout<<"AT EVAN 1..."<<endl;

  using TfLiteDelegateCreateFn = TfLiteDelegate* (*)();
  TfLiteDelegateCreateFn create_delegate = reinterpret_cast<TfLiteDelegateCreateFn>(
      dlsym(delegate_lib, "tflite_plugin_create_delegate")); // Replace "CreateDelegate" with the actual function name

  if (!create_delegate) {
      std::cerr << "Failed to get the delegate creation function: " << dlerror() << std::endl;
  }

  cout<<"AT EVAN 2..."<<endl;

  auto delegate = create_delegate();

  cout<<"AT EVAN 3..."<<endl;

  dlclose(delegate_lib);

  cout<<"AT EVAN 4..."<<endl;
  */
  
  // tflite::StatefulNnApiDelegate::Options options;
  // auto delegate = tflite::evaluation::CreateNNAPIDelegate(options);
  // if (!delegate)
  // {
  //   cout << endl
  //        << endl
  //        << "ERROR IN MAKING THE DELEGATE FOR NNAPI!!!" << endl
  //        << endl;
  //   exit(-1);
  // }
  // else
  // {
  //   interpreter->ModifyGraphWithDelegate(std::move(delegate));
  //   cout << endl
  //        << endl
  //        << "SUCCESSFULLY ADDED NNAPI DELEGATE TO INTERPRETER..." << endl
  //        << endl;
  // }
  //  END OF TEST ADDING

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
    img = cv::imread(infile);
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

  cv::Mat inputImg = mat_process(img, WIDTH, HEIGHT); // could be input is modified by this function
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
  float MAX_ACCEPT_SCORE = 0.25;
  float bbox_score = 1;

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
  printf("%f ", output1[8400 * 1]);
  printf("%f ", output1[8400 * 2]);
  printf("%f ", output1[8400 * 3]);
  printf("%f ", output1[8400 * 4]);
  cout << " done print outputs " << endl;

  cout << " dim1 is " << dim1 << endl;
  // 705600
  for (int bdim = 0; bdim < dim1; bdim++)
  {

    for (int oi = 4; oi < dim2; oi++)
    {
      // printf("%f ", output1[(dim1 * oi) + bdim]);
      float test_score = output1[(dim1 * oi) + bdim];
      if (test_score > max_class_score)
      {
        max_class_score = test_score;
        max_class = oi - 4;
      }
    }

    // printf("\n");
    // cout << max_class_score << " " << max_class << endl;
    if (max_class_score > MAX_ACCEPT_SCORE)
    {
      // printf("\n");
      // for(int oj=oi-85; oj<oi; oj++){
      //   printf("%f ", output1[oj]);
      // }
      // printf("\n");
      const float cx = output1[bdim];              // output1[oi-84];
      const float cy = output1[(dim1 * 1) + bdim]; // output1[oi-83];
      const float w = output1[(dim1 * 2) + bdim];  // output1[oi-82];
      const float h = output1[(dim1 * 3) + bdim];  // output1[oi-81];
      cout << " xywh HEIGHT WIDTH COLS ROWS " << cx << " " << cy << " " << w << " " << h << " " << HEIGHT << " " << WIDTH << " " << img.cols << " " << img.rows << endl;
      const float xmin = ((cx - (w / 2))) * img.cols / 640;
      const float ymin = ((cy - (h / 2))) * img.rows / 640;
      const float xmax = ((cx + (w / 2))) * img.cols / 640;
      const float ymax = ((cy + (h / 2))) * img.rows / 640;
      rects.emplace_back(cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin));
      cout << xmin << " " << ymin << " " << xmax - xmin << " " << ymax - ymin << endl;
      vector<float> currvec = {xmin, ymin, xmax, ymax};
      recvec.emplace_back(currvec);
      // rects.emplace_back(cv::Rect((output1[oi-85]-(0.5*output1[oi-83]))*img.cols, (output1[oi-84]-(0.5*output1[oi-82]))*img.rows, output1[oi-83]*img.cols, output1[oi-82]*img.rows));
      scores.emplace_back(max_class_score);
      // ids.emplace_back(max_class);
      // cout << (output1[oi-85]-(0.5*output1[oi-83]))*img.cols << " " << (output1[oi-84]-(0.5*output1[oi-82]))*img.rows << " " << output1[oi-83]*img.cols << " " << output1[oi-82]*img.rows << " " << max_class << " " << max_class_score << endl;
      // cout << output1[oi-85] << " " << output1[oi-84] << " " << output1[oi-83] << " " << output1[oi-82] << " " << endl;
      cout << " MAX CLASS  MAX SCORE " << max_class << " " << max_class_score << endl;
      cout << " nelem is " << nelem << endl;
      // cout << " BBOX SCORE IS " << output1[oi-86] << endl;
      // cout << " FIRST CLASS VALUE IS " << output1[oi-85] << endl;

      if (map_rects.find(max_class) == map_rects.end())
      {
        map_rects[max_class] = vector<vector<float>>();
        map_rects[max_class].emplace_back(currvec);
        map_scores[max_class] = vector<float>();
        map_scores[max_class].emplace_back(max_class_score);
        // not found
      }
      else
      {
        // found
        map_rects[max_class].emplace_back(currvec);
        map_scores[max_class].emplace_back(max_class_score);
      }
    }

    max_class_score = 0;
    max_class = -1;
    bbox_score = 1;
    // oi += 4;
  }

  done = std::chrono::system_clock::now();
  elapsed_seconds = done - end;
  printf("Create Boxes from interpreter outputs s: %.10f\n", elapsed_seconds.count());

  // cv::dnn::NMSBoxes(rects,scores,0.2,0.4,nms);
  // vector<int> pick;
  /*vector<Rect>*/
  int prev_count = 0;
  for (auto element : map_rects)
  {
    // int class_id = element.first;
    // nmsrec.emplace_back(nms_vec(element.second,0.5, pick));
    vector<Rect> temp_nms_rec;
    vector<int> temp_pick;
    cout << "1 Size of temp nms rec is " << temp_nms_rec.size() << endl;
    cout << "1 Size of temp pick is " << temp_pick.size() << endl;
    cout << "1 Size of nms rec is " << nmsrec.size() << endl;
    cout << "1 Size of vec rect is " << element.second.size() << endl;
    temp_nms_rec = nms_vec(element.second, 0.5, temp_pick);

    copy(temp_nms_rec.begin(), temp_nms_rec.end(), back_inserter(nmsrec));
    cout << "2 Size of temp nms rec is " << temp_nms_rec.size() << endl;
    cout << "2 Size of temp pick is " << temp_pick.size() << endl;
    cout << "2 Size of nms rec is " << nmsrec.size() << endl;
    cout << "2 Prev count is " << prev_count << endl;
    for (int index = 0; index < temp_pick.size(); index++)
    {
      // cout << " Temp pick value is " << temp_pick[index] << endl;
      cout << " Emplace back with value " << index + prev_count << endl;
      pick.emplace_back(index + prev_count);
    }
    // copy(temp_pick.begin(), temp_pick.end(), back_inserter(pick));
    ids.insert(ids.end(), temp_nms_rec.size(), element.first);
    prev_count += temp_nms_rec.size();
    cout << " Size of Ids is " << ids.size() << endl;
    // ids.emplace_back()
    // nmsrec.emplace_back(temp_nms_rec);
    // pick.emplace_back(temp_pick);
  }

  // nmsrec = nms_vec(recvec,0.5, pick);

  nmsdone = std::chrono::system_clock::now();
  elapsed_seconds = nmsdone - done;
  printf("Process nms s: %.10f\n", elapsed_seconds.count());

  cv::Mat clnimg = img.clone();
  int pick_index = 0;
  for (cv::Rect rect : nmsrec)
  {
    cv::rectangle(clnimg, rect, cv::Scalar(0, 255, 0), 3);
    cv::putText(clnimg, to_string(ids[pick[pick_index]]), cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(118, 185, 0), 2);
    cout << " New rect is " << rect << endl;
    cout << " New index is " << pick[pick_index] << endl;
    cout << " Class is " << ids[pick[pick_index]] << endl;
    pick_index++;
  }

  cv::imwrite(outfile, clnimg);
  // cout << " WROTE IMAGE " << endl;

  // printf("Result is: %f\n", *output2);

  // std::cout << "OUTPUT BOX DIMS ARE " << output_box->dims->size << "\n"; // Tensor<type: float shape: [] values: 30>
  // std::cout << "OUTPUT BOX DIMS ARE " << output_box->dims->data << "\n"; // Tensor<type: float shape: [] values: 30>
  // std::cout << "OUTPUT SCORE DIMS ARE " << output_score->dims << "\n"; // Tensor<type: float shape: [] values: 30>

  /*
  vector<float> box_vec = cvtTensor(output_box);
  //vector<float> score_vec = cvtTensor(output_score);
  cout << " GOT CVT TENSOR " << endl;

  vector<size_t> result_id;
  auto it = std::find_if(std::begin(score_vec), std::end(score_vec),
                         [](float i){return i > 0.6;});
  while (it != std::end(score_vec)) {
      result_id.emplace_back(std::distance(std::begin(score_vec), it));
      it = std::find_if(std::next(it), std::end(score_vec),
                        [](float i){return i > 0.6;});
  }

  cout << " GOT RESULT ID " << endl;

  vector<cv::Rect> rects;
  vector<float> scores;
  for(size_t tmp:result_id){
      const int cx = box_vec[4*tmp];
      const int cy = box_vec[4*tmp+1];
      const int w = box_vec[4*tmp+2];
      const int h = box_vec[4*tmp+3];
      const int xmin = ((cx-(w/2.f))/WIDTH) * img.cols;
      const int ymin = ((cy-(h/2.f))/HEIGHT) * img.rows;
      const int xmax = ((cx+(w/2.f))/WIDTH) * img.cols;
      const int ymax = ((cy+(h/2.f))/HEIGHT) * img.rows;
      rects.emplace_back(cv::Rect(xmin, ymin, xmax-xmin, ymax-ymin));
      scores.emplace_back(score_vec[tmp]);
  }

  cout << " GOT RECTS " << endl;

  vector<int> ids;

  //cv::dnn::NMSBoxes(rects, scores, 0.6, 0.4, ids);
  //for(int tmp: ids)
  //   cv::rectangle(img, rects[tmp], cv::Scalar(0, 255, 0), 3);
  for(cv::Rect rect: rects){
      cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 3);
  }

  cout << " GOT CV RECT " << endl;
  cv::imwrite("./result.jpg", img);
  cout << " WROTE IMAGE " << endl;
  */

  return inputImg;
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
  string imgf1 = "image0frame0.jpg";
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
  char *img2 = read_image(imgf2, filesize);
  cv::Mat img2_mat = convert_image(img2, 1080, 1920, filesize);

  // cv::Mat img1_comp = imread(imgf1);

  // cout << " Start process " <<endl;

  // cv::Mat conv1 = mat_process(img1_comp,320,320);
  // cout << " Done 1 " << endl;
  // cv::Mat conv2 = mat_process(img1_mat,320,320);
  // cout << " Done process " <<endl;

  // cout << "Channels are: " << img1_mat.channels() << " Size is: " << img1_mat.size() << " Flags: " << img1_mat.flags << " Cols " << img1_mat.cols << " Rows: "<< img1_mat.rows << endl;
  // cout << "Channels are: " << img1_comp.channels() << " Size is: " << img1_comp.size() << " Flags: " << img1_comp.flags << " Cols " << img1_comp.cols<< " Rows: "<< img1_comp.rows << endl;
  // bool is_equal = (sum(img1_mat!= img1_mat) == Scalar(0,0,0,0));
  // cout << " Equality check was " << is_equal << endl;

  cv::Mat pmat1 = process_3("", "./result0.jpg", nmsrec1, pick1, ids1, &img1_mat);

  cv::Mat pmat2 = process_3("", "./result1.jpg", nmsrec2, pick2, ids2, &img2_mat);
  return 0;
  // string test_file = "viper_snapshot.jpg";
  // char* img_test = read_image(test_file);
  // cv::Mat result = convert_image(img_test);
  // cv::Mat pmat3 = process_2(test_file, "./result3.jpg", nmsrec3, pick3, ids3, &result);
  // cout << "Processed the result " << endl;

  // cv::Mat i1 = cv::imread(imgf1);
  // cv::Mat i2= mat_process(i1, 1920, 1080); //could be input is modified by this function
  // cv::Mat j1 = cv::imread(imgf2);
  // cv::Mat j2= mat_process(j1, 1920, 1080); //could be input is modified by this function

  // char* img1 = read_image(imgf1);
  // char* img2 = read_image(imgf2);
  std::cout << " Got images " << endl;
  // cv::Mat mat1 = get_mat(img1);
  // cv::Mat mat2 = get_mat(img2);
  cout << " Starting motion detect " << endl;
  motion_detect(img1_mat, img2_mat, nmsrec2, pick2, ids2, motion2);
  cout << "Processing motion detection " << endl;

  cout << " Deleting malloc image " << endl;
  // delete img1;
  // delete img2;
  cout << " Done Delete " << endl;

  auto t1 = chrono::high_resolution_clock::now();

  // char * img1 = read_image(imgf1, filesize);
  Mat imgv_mat = convert_image(img1, 1080, 1920, filesize);
  cv::Mat out_t, outt1, outt2, outt3, outt4;
  cv::Scalar mat_mean, mat_stdev;
  absdiff(img1_mat, img2_mat, out_t);
  double total_mean = 0;
  double total_stdev = 0;
  cv::meanStdDev(out_t, mat_mean, mat_stdev);
  // mat_mean = cv::sum(dst);
  // cout << " THE MAT_MEAN2 WAS..............................." << mat_mean[0] << endl;
  for (int i = 0; i < 3; i++)
  {
    total_mean += mat_mean[i];
    total_stdev += mat_stdev[i];
    cout << " Total mat mean " << mat_mean[i] << endl;
    cout << " Total mean " << total_mean << " " << total_stdev << endl;
  }

  threshold(out_t, outt1, 0, 255, 3);
  auto t2 = chrono::high_resolution_clock::now();

  cv::Mat avg = 0.9 * out_t + 0.1 * imgv_mat;
  auto t3 = chrono::high_resolution_clock::now();

  cv::imwrite("./outt1.jpg", outt1);

  threshold(out_t, outt2, 50, 255, 3);
  cv::imwrite("./outt2.jpg", outt2);

  threshold(out_t, outt3, 75, 255, 3);
  cv::imwrite("./outt3.jpg", outt3);

  threshold(out_t, outt4, 200, 255, 3);
  cv::imwrite("./outt4.jpg", outt4);

  std::chrono::duration<double> elapsed_seconds = t2 - t1;
  std::chrono::duration<double> elapsed_seconds2 = t3 - t2;

  printf("The time for motion diff was s: %.10f\n", elapsed_seconds.count());
  printf("The time to compute average matrix was s: %.10f\n", elapsed_seconds2.count());
}