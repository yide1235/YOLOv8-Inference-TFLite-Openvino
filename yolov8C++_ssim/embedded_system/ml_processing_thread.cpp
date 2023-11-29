#include <iostream>
#include <unistd.h>
#include <gst/gst.h>
#include <gst/app/app.h>
//ns
//#define STB_IMAGE_IMPLEMENTATION
//ns

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


#include "include/threading/ml_processing_thread.hpp"
#include "include/threading/server.hpp"
#include "include/map_set.h"
#include "include/debug_log.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>


#include "include/util/stb_image.h"
#include "include/util/stb_image_write.h"
#include "include/util/stb_image_resize.h"
#include "include/util/nms.hpp"
#include "include/util/nms_utils.hpp"
#include "include/util/ml_image_manager.hpp"


//#include <cairo/cairo.h>

using namespace std;
using namespace tflite;
using namespace cv;


using TfLiteDelegatePtr = tflite::Interpreter::TfLiteDelegatePtr;
//ns
using TfLiteDelegatePtrMap = std::map<std::string, TfLiteDelegatePtr>;


mutex frame_mutex;
int PROCESS_RATE = 1; //after how many images to process a batch
//int PREV_COUNT = 0; //what was the previous size of the queue
//float MOD_VAL = 1; //after how many frames to add to the queue for example 6 means every 6 frames 1 frame is added
int NUM_COUNT = 0;  // how many iterations have happened
//float INIT_MOD_VAL = 3; //the initial mod val to keep track of
//int PREV_COUNT = 0;  //what was the size in the queue
//int QUEUE_RATE = 0; //what is the average rate of change of the queue
int MAX_DESIRED_QUEUE_SIZE = 20; //don't want the queue to fill up with more than 20 ideally

long IMAGE_AVG = 0; //the running average of the image
long IMAGE_STD_DEV = 1; //the running std dev of the image
//long IMAGE_CTR = 0;  //a counter to determine if we have enough data 
//int IMAGE_CTR_INIT = 50; //how many data points before the counter is initialized
float IMAGE_LEARNING_RATE = 0.02; //how much to affect the running avg and std dev with the current learning
//float IMAGE_NUM_STDDEV_MOTION = 1; //how many standard deviations away to accept motion
//map<int,float> IMAGE_RUNNING_COUNT_GOOD;
//map<int,float> IMAGE_RUNNING_COUNT_BAD; 
float IMAGE_RL_STD_DEV[81] = {0}; //initialize the RL controller to 0
//double MOTION_CUTOFF = 50; //value used to determine if there is motion or not 

Mat imgv_mat_last; //the last frame to compare to

mutex avg_mat_mutex;
Mat imgv_mat_avg; //the running average of the frame

//vector<map<int,string>> ZONES;   
//ns
#define LOG(severity) (std::cerr << (#severity) << ": ");
bool keep_ml_thread_alive(){
    ss_mutex.lock();
    bool keep_alive = !serv_state.shutting_down && !serv_state.change_http_conf && !serv_state.network_config_failure;
    ss_mutex.unlock();

    return keep_alive;
}

viper_image::viper_image(unsigned char* img, int img_size, guint64 frame_time, guint64 segment_time, std::tm localtime, int segment_id, int width, int height, bool motion_flag, int rl_index){
    this->img = img;
    this->img_size = img_size;
    this->frame_time = frame_time;
    this->segment_time = segment_time;
    this->localtime = localtime;
    this->segment_id = segment_id;
    this->width = width;
    this->height = height;
    this->motion_flag = motion_flag;
    this->rl_index = rl_index;
}

viper_image::~viper_image(){
    //cout << "Deleting viper image " << to_string(segment_id) << endl;
    //delete img;
}

viper_zone::viper_zone(vector<float> vertx, vector<float> verty, int vert, string zone_id, string zone_x, string zone_y){
  this->vert = vert;
  this->vertx = vertx;
  this->verty = verty;
  this->zone_id = zone_id;
  this->zone_x = zone_x;
  this->zone_y = zone_y;
}

viper_zone::~viper_zone(){
    //cout << "Deleting viper image " << to_string(segment_id) << endl;
    //delete img;
}

viper_sequence::viper_sequence(int seq, string image_path, int frameseq, string localtime_str){
  this->seq = seq;
  this->image_path = image_path;
  this->frameseq = frameseq;
  this->localtime_str = localtime_str;
}

viper_sequence::~viper_sequence(){

}

int binary_search(std::deque<viper_sequence> sequences, std::tm value) {
    
    int low = 0;
    int high = sequences.size() - 1;

    while (low <= high) {
        
        int mid = low + (high - low) / 2;
        cout << " comparing " << sequences[mid].localtime_str << " " << mid << " " << low << " " << high << " " << endl;
        std::tm comp = str_to_datetm(sequences[mid].localtime_str);
        if (std::mktime(&comp) < std::mktime(&value)) {
              low = mid + 1;
              cout << " comp is less than reference " << endl;
        } else {
              high = mid - 1;
              cout << " comp is more than reference " << endl;
        }
        
    }

    return low;
}

tuple<int,int,string> search_events_by_datetime(std::tm start_date, std::tm end_date) {
    lock_guard<mutex> image_guard(std_ml_image_event_mutex);
    std::time_t time_value = std::mktime(&end_date);
    // Add 1 second to the time_t value
    time_value += 1;
    end_date = *std::localtime(&time_value);

    int start_index = binary_search(std_ml_image_event_deque, start_date);
    cout << " start index for start date is " << start_index << endl;

    if(start_index > std_ml_image_event_deque.size()-1){
      return make_tuple(-1,0,"");
    }

    int end_index = binary_search(std_ml_image_event_deque, end_date);
    if(end_index == 0){
      return make_tuple(0,-1,"");
    }
    cout << " end index for end date is " << end_index << endl;



    if(end_index > 0){
      end_index--;
    }
    cout << " end index after adjustment for end date is " << end_index << endl;

    string resp = "";
    for(int i=start_index; i<=end_index; i++){
      resp += to_string(std_ml_image_event_deque[i].seq) + ",";
    }

    return make_tuple(std_ml_image_event_deque[start_index].seq, std_ml_image_event_deque[end_index].seq, resp);


    //void get_image_events(int refseq, int num, bool seek_newest, vector<string>& seqlist, vector<string>& images, vector<string>& meta, vector<string>& frameseq, vector<string>& image_names){

}


int horz_intersect(float x1, float y1, float x2, float y2, float testx, float testy) {
    // y = mx + b
    float m = (y2 - y1) / (x2 - x1);
    float b = y1 - m * x1;
    //horizontal line is y = testy     y = mx + b         testy = mx+b   x = testy-b / m
    float checkx = (testy - b) / m;
    if (checkx < testx) { return 0; } //checking for intersection to the right only
    if (checkx == x1 || checkx == x2) { return 1; } //of the intersection is at the point have to make a further check

    if (x1 > x2) {
        if (checkx > x2 && checkx < x1) {
            return 1;
        }
        else {
            return 0;
        }
    }
    else {
        if (checkx > x1 && checkx < x2) {
            return 1;
        }
        else {
            return 0;
        }
    }
}

int pnpoly(int nvert, float* vertx, float* verty, float testx, float testy)
{//vertices must be in clockwise order to be able to convert to line seqments in pair X,X+1,
    int res = 0;
    for (int i = 0; i < nvert; i++) {
        //cout << i << endl;
        float x1 = *vertx;
        float y1 = *verty;
        if (i == nvert - 1) {
            vertx = vertx - (nvert - 1);
            verty = verty - (nvert - 1); //take the very first value for the last segment
        }
        else {
            vertx++;
            verty++;
        }
        //std::cout << " test values are " << x1 << " " << y1 << " " << *vertx << " " << *verty << " " << testx << " " << testy << std::endl;
        res+=horz_intersect(x1, y1, *vertx, *verty, testx, testy);
    }
    //cout << " res before modulo is " << res << endl;
    return res % 2;
}

vector<float> get_floats(string s){
  vector<float> v;
  std::stringstream ss(s);
  while(ss.good()){
    try{
      string substr;
      getline(ss,substr,',');
      float f = std::stof(substr);
      v.push_back(f);
      //cout << " Got the float for vertex " << f << endl;
    }catch(std::exception & e){

    }
  }
  return v; 
}

bool check_load_zones(){

  //init the zones if not yet done
  if(!std_ml_camera_zones_init_state){
    string zonestr = get_setting("ZoneSettings", "NumZones", false);
    int zones = std::stoi(zonestr);
    for(int i=0; i<zones; i++){
      string valx = "Z" + to_string(i+1) + "X";
      string valy = "Z" + to_string(i+1) + "Y";
      string valid = "Z" + to_string(i+1) + "ID";
      string valact = "Z" + to_string(i+1) + "IsActive";
      //string valclass = "Z" + to_string(i+1) + "Classes";

      string isactive = get_setting("ZoneSettings", valact, false);
      if(toLowerCase(isactive) != "true"){
        continue;
      }

      string id = get_setting("ZoneSettings", valid, false);
      string x = get_setting("ZoneSettings", valx, false);
      string y = get_setting("ZoneSettings", valy, false);
      
      vector<float> fx = get_floats(x);
      vector<float> fy = get_floats(y);
      if(fx.size() != fy.size()){
        continue;
      }else{
        viper_zone zone(fx, fy, (int)fx.size(), id,x,y);
        std_ml_camera_zones_deque.push_back(zone);
        std_ml_zone_def_str += "[" + id + ", X:[" + x + "], " +  "Y:[" + y + "]], ";       
      }

      /*
      //zone settings are disabled for the class because they are maintained at the user level      
      string zone_classes  = get_setting("ZoneSettings", valclass, false);
      vector<float> zc = get_floats(zone_classes);
      map<int,string> curr_zone;
      if(zc.size()==0){ //if empty all classes are valid for the zone
        string num_classes = get_setting("DetectionSettings", "NumSettings", false);
        int num_cls = stoi(num_classes);
        for(int j=0; j<num_cls;j++){
          curr_zone.insert(make_pair(j+1,"Z"+to_string(i+1)));
        }

      }else{
        for(float curr_class: zc){
          curr_zone.insert(make_pair((int)curr_class,"Z"+to_string(i+1)));
        }
      }
      ZONES.push_back(curr_zone);*/
    }//end for

    std_ml_zone_def_str = std_ml_zone_def_str.substr(0,std_ml_zone_def_str.length()-2); //removes the trailing commas in the zone formatting


    std_ml_camera_zones_init_state = true;
  }
  //end init zones
  return true;
}



vector<string> check_zones(cv::Rect test_rec){
  check_load_zones();
  vector<string> zone_intersections;
  float p1x[8];
  float p1y[8];
  p1x[0] = test_rec.x;  //top left 
  p1y[0] = test_rec.y;

  p1x[1] = test_rec.x + (test_rec.width/2);  //top center
  p1y[1] = test_rec.y;

  p1x[2] = test_rec.x;
  p1y[2] = test_rec.y + (test_rec.height/2); //mid left

  p1x[3] = test_rec.x + test_rec.width; //top right
  p1y[3] = test_rec.y;

  p1x[4] = test_rec.x; //bottom left
  p1y[4] = test_rec.y + test_rec.height;

  p1x[5] = test_rec.x + (test_rec.width/2); //bottom mid
  p1y[5] = test_rec.y + test_rec.height;

  p1x[6] = test_rec.x + (test_rec.width); //mid right
  p1y[6] = test_rec.y + (test_rec.height/2);

  p1x[7] = test_rec.x + (test_rec.width); //bottom right
  p1y[7] = test_rec.y + (test_rec.height);


  for(int i=0; i<std_ml_camera_zones_deque.size(); i++){
      int res = 0;
      //check if the bounding box vertices are in the zone
      #ifdef VIPER_DEBUG_ENABLED
        std::cout << " Checking zone " << i << " in the list for 8 vertices " << endl;
      #endif
      for(int j=0; j<8; j++){
         res = pnpoly(std_ml_camera_zones_deque.at(i).vert, &std_ml_camera_zones_deque.at(i).vertx[0], &std_ml_camera_zones_deque.at(i).verty[0], p1x[j], p1y[j]);
         if(res == 1){
          zone_intersections.push_back(std_ml_camera_zones_deque.at(i).zone_id);
          break;
         }
      }

      if(res == 0){
        #ifdef VIPER_DEBUG_ENABLED
          std::cout << " Didn't find anything checking if any zone points inside the box " << endl;
        #endif
        float *x = &std_ml_camera_zones_deque.at(i).vertx[0];
        float *y = &std_ml_camera_zones_deque.at(i).verty[0];
        //check if the zone vertices are in the bounding box
        for(int j=0; j<std_ml_camera_zones_deque.at(i).vert; j++){

          x++;
          y++;
          res = pnpoly(8, &p1x[0], &p1y[0], *x, *y);
          if(res == 1){
            zone_intersections.push_back(std_ml_camera_zones_deque.at(i).zone_id);
            break;
          }
        }
      }
      
  }

  return zone_intersections;
}

long avg_array(unsigned char* img, int img_size){
  long sum = 0;
  for(int i=0; i<img_size; i++){
    sum += (long)*(img+i);
  }

  return sum;
  /*if(img_size > 0){
    return sum/(long)img_size;
  }else{
    return 0;
  }*/

}

long std_dev(long CURR_AVG){
  //variance is 1/n * sum(xi-xbar)
  /*long new_stdev = (CURR_AVG - IMAGE_AVG) * (CURR_AVG - IMAGE_AVG);
  new_stdev = sqrt(new_stdev);
  new_stdev *= IMAGE_LEARNING_RATE;
  new_stdev += IMAGE_STD_DEV * (1-IMAGE_LEARNING_RATE);
  return new_stdev;*/
  long cum = 1/(IMAGE_LEARNING_RATE);
  //cout << " CUM IS " << to_string(cum) << endl;
  long new_stdev = ((long)IMAGE_STD_DEV * (long)IMAGE_STD_DEV * (cum-1) ) ;
  //cout << " std dev " << new_stdev << endl;
  new_stdev += ( ( (long)CURR_AVG - (long)IMAGE_AVG ) * ( (long)CURR_AVG - (long)IMAGE_AVG ) );
  //cout << " std dev " << new_stdev << endl;
  new_stdev = sqrt(new_stdev / cum);
  //cout << " std dev " << new_stdev << endl;
  return new_stdev;
}

bool check_rl_controller(float run_std_dev, float curr_std_dev, float run_avg, float curr_avg, int& index){
  try{

    float num_stdevs = sqrt(pow((curr_avg - run_avg) / (run_std_dev+1),2)  +  pow(curr_std_dev / (run_std_dev+1),2));   //calculate how many standard deviations is the difference
    /*if(std_dev == 0){ std_dev = 0.001;} //prevent division by zero

    float num_stdevs = (curr_avg - run_avg) / std_dev;   //calculate how many standard deviations is the difference
    */
    int int_stdevs = (int)(num_stdevs); // this should be around 1-3 for non-motion  and greater than 4 for motion
    if(int_stdevs > 40){ int_stdevs = 4;}  //if more than -40 std devs it is added to the end bucket
    else if(int_stdevs < -4) { int_stdevs = -4;} //if more 40 std devs it is added to the end bucket

    index = 4 + int_stdevs; //-4 = 0, 4=80   get the index in the RL ARRAY

    //cout << " Num std devs is " << to_string(num_stdevs) << " RL val is " << to_string(IMAGE_RL_STD_DEV[index]) << " run std dev is " 
    //<< to_string(run_std_dev) << " curr std dev is " << to_string(curr_std_dev) << " run avg " 
    //<< to_string(run_avg) << " curr avg " << to_string(curr_avg) << endl;

    if(IMAGE_RL_STD_DEV[index] > 0){  // if the value is more than 1 it means there is a good chance of finding motion
      return true;
    }else{
      if(IMAGE_RL_STD_DEV[index] == 0 && num_stdevs > 4.0){
        return true;
      }else{
        return false;
      }
    }

  }catch(exception &e){
    index = -1;
    return false;
  }

}

void update_rl_controller(int index, bool is_motion, bool motion_found){
  //this update parameter is weighted towards keeping queue size small and finding true motion events that the 
  //random image processing does not find
  #ifdef VIPER_DEBUG_ENABLED
    std::cout << " Index is " << to_string(index) << " is_motion is " << to_string(is_motion) << " motion found is " << to_string(motion_found) << endl;
  #endif

  if(index <0 || index >80) { return; } //invalid index do nothing
  //float temp_mod_val = MOD_VAL;
  //if(temp_mod_val <= 0){ temp_mod_val = 1;}

  //std::cout << " Temp mod val is " << to_string(temp_mod_val) << endl;

  float update_val = 0;
  if(is_motion && motion_found){
      #ifdef VIPER_DEBUG_ENABLED
        std::cout << " case 1 " << endl;
      #endif
      update_val = 1.0 / (1.0 + std_ml_image_proc_deque.size()); //if motion is found the reward parameter is the inverse of the queue size
  }else if(is_motion && !motion_found){
      #ifdef VIPER_DEBUG_ENABLED    
        std::cout << " case 2 " << endl;
      #endif
      update_val = -1/MAX_DESIRED_QUEUE_SIZE; //if motion not found subtract 1/MAX_DESIRED_QUEUE_SIZE
  }else if(!is_motion && motion_found){
      #ifdef VIPER_DEBUG_ENABLED
      std::cout << " case 3 " << endl;
      #endif
      update_val =  1.0 / (1.0 + std_ml_image_proc_deque.size());  //if motion is found without the motion tracker then reward is inverse of the queue size times 1 / MODVAL, lesser reward 
  }else{
      #ifdef VIPER_DEBUG_ENABLED
        std::cout << " case 4 " << endl;
      #endif
      update_val = 0; //if motion is not found without the motion tracker, then we do nothing
  }
  #ifdef VIPER_DEBUG_ENABLED
    std::cout << " Update val is " << to_string(update_val) << endl;
  #endif
  //update the matrix with the discount parameter
  IMAGE_RL_STD_DEV[index] = IMAGE_RL_STD_DEV[index] * (1-IMAGE_LEARNING_RATE) + update_val * (IMAGE_LEARNING_RATE);
  #ifdef VIPER_DEBUG_ENABLED
    std::cout << " The RL Value is " << to_string(IMAGE_RL_STD_DEV[index]) << endl;
  #endif
}


cv::Mat convert_image(char* img, int height, int width, int filesize){
  cv::Mat mat_img;

  mat_img = cv::imdecode(cv::Mat(1, filesize, CV_8UC1, img), IMREAD_UNCHANGED);
  return mat_img;
}

void ml_load_image(unsigned char* img, int img_size, guint64 frame_time, guint64 segment_time, std::tm localtime, int segment_id , int width, int height){
    
    bool bypass = false; //bypass does a quick statistical check for motion to ensure good frames don't get dropped;
    Mat imgv_mat = convert_image((char*) img, height, width, img_size);  

    //save the image here
    
    if(std_videos_img_counter % std_videos_save_frames == 0){
        std_videos_img_counter = 0;
        string img_path = std_videos_save_dir + std_videos_image_file_name;
        #ifdef VIPER_DEBUG_ENABLED
          std::cout << "Save image path is......" << img_path ;
        #endif
        lock_guard<mutex> video_guard(std_videos_img_mutex);
        //ifstream f(img_path.c_str());
        cv::imwrite(img_path, imgv_mat, {cv::IMWRITE_JPEG_QUALITY,std_videos_jpeg_quality});
    }else{
      std_videos_img_counter++;
    } 
    
    return;


    
    int index = -1;
    //float num_std = 0; //number of std_deviations from the average
    NUM_COUNT++;

    if(imgv_mat_last.empty()){

      imgv_mat_last = imgv_mat;
      imgv_mat_avg = imgv_mat;

    }else{

      try{

        if(NUM_COUNT % 5 && avg_mat_mutex.try_lock()){
          imgv_mat_avg = (1-IMAGE_LEARNING_RATE) * imgv_mat_avg  + (IMAGE_LEARNING_RATE) * imgv_mat; //add this frame to the running average
          //cout << " Updated the img_mat_avg Matrix " << endl;
          avg_mat_mutex.unlock();
          //cout << " Unlocked the  avg_mat_mutex" << endl;
        }
      }catch(exception &e1){
        if(avg_mat_mutex.try_lock());
        avg_mat_mutex.unlock();
      }

      cv::Mat out_t, outt1, outt2, outt3, outt4;
      cv::Scalar mat_mean, mat_stdev;

      
      absdiff(imgv_mat,imgv_mat_last, out_t);
      threshold(out_t, outt2, 25, 255, 3);
      double total_mean=0;
      double total_stdev = 0;
      cv::meanStdDev(outt2,mat_mean,mat_stdev);

      imgv_mat_last = imgv_mat; //set the last mat to be the curr mat

      long CURR_AVG = 0;
      long NEW_STDEV = 0;
      //mat_mean = cv::sum(dst);
      //cout << " THE MAT_MEAN2 WAS..............................." << mat_mean[0] << endl;
      for(int i=0; i<3; i++){
        CURR_AVG += mat_mean[i];
        NEW_STDEV += mat_stdev[i];
        //cout << " Total mat mean " << mat_mean[i] << endl;
      }
      //cout << " Total mean, stddev: " << CURR_AVG  << " , " << NEW_STDEV<< endl;


      
      //cout << " CHECK MOTION " << to_string(CURR_AVG) << " NEW STD DEV" << to_string(NEW_STDEV) << " IMAGE AVG " << to_string(IMAGE_AVG) << "  NUM STDEVS MOTION: " << to_string(IMAGE_LEARNING_RATE) << endl;
      //CHECK MOTION 123 NEW STD DEV44 IMAGE AVG 76  NUM STDEVS MOTION: 0.020000
      bypass = check_rl_controller(1, NEW_STDEV, 1, CURR_AVG, index); //this is set to 5/5 for now
    

      if(bypass){
        #ifdef VIPER_DEBUG_ENABLED
          std::cout << " CHECK MOTION " << to_string(CURR_AVG) << " NEW STD DEV" << to_string(NEW_STDEV) << " IMAGE AVG " << to_string(IMAGE_AVG) << "  NUM STDEVS MOTION: " << to_string(IMAGE_LEARNING_RATE) << endl;
          std::cout << " BY PASS WAS " << bypass << endl;
        #endif
      }
   
      IMAGE_AVG = (1-IMAGE_LEARNING_RATE)*IMAGE_AVG + IMAGE_LEARNING_RATE * CURR_AVG;
      IMAGE_STD_DEV = (1-IMAGE_LEARNING_RATE)*IMAGE_STD_DEV + IMAGE_LEARNING_RATE * NEW_STDEV ;

    }

    if(!bypass){
      delete [] img; //no image then nothing to process
      return;
    }
      
    

    
    {//start the lock_guard scope
      //std_ml_image_proc_deque_preprocess
      int new_segment_id = 0;
      guint64 new_time=0;
      guint64 new_segment_time=0;
      if(ml_image_mgr_check_segment(segment_id, frame_time, new_segment_id , new_time, new_segment_time)){
        viper_image* newframe =  new viper_image(img,img_size,new_time,new_segment_time,localtime, segment_id, width, height, bypass, index);
        lock_guard<mutex> frame_guard(frame_mutex);
        std_ml_image_proc_deque.push_back(newframe);
      }else{

        //img needs to be deleted when used up
        viper_image* newframe =  new viper_image(img,img_size,frame_time,segment_time,localtime, segment_id, width, height, bypass, index);
        lock_guard<mutex> frame_guard(frame_mutex);
        std_ml_image_proc_deque_preprocess.push_back(newframe);
      }
    }//end the lock_guard scope
    #ifdef VIPER_DEBUG_ENABLED
      std::cout << "Pushing viper_image " << to_string(segment_id) << " with size " << std_ml_image_proc_deque.size() << endl;
    #endif
    
    

}

void check_move_images_from_preprocessing(){
  //images are in the preprocess queue
  lock_guard<mutex> frame_guard(frame_mutex);
  //cout << " Checking image manager for processing " << endl;
  int i=0;
  while(true){
    if(std_ml_image_proc_deque_preprocess.size() > 0){
      viper_image* newframe = std_ml_image_proc_deque_preprocess.at(0);
      int seq = newframe->segment_id;
      guint64 time = newframe->frame_time;
      int newseq = 0;
      guint64 newtime=0;
      guint64 newsegmenttime=0;
      if(ml_image_mgr_check_segment(seq, time, newseq, newtime, newsegmenttime) && i < 5){
          std_ml_image_proc_deque_preprocess.pop_front();
          newframe->segment_id = newseq;
          newframe->frame_time = newtime;
          newframe->segment_time = newsegmenttime;
          std_ml_image_proc_deque.push_back(newframe);
          #ifdef VIPER_DEBUG_ENABLED
            cout << "Moving frames to the processing queue " << to_string(std_ml_image_proc_deque.size()) 
                        << " " << to_string(std_ml_image_proc_deque_preprocess.size()) << endl;
          #endif
          i++;
      }else{
          #ifdef VIPER_DEBUG_ENABLED
            cout << " Done preprocessing queue " << endl;
          #endif
          return;
      }
    }else{
      return;
    }
  }

}

vector<int> sort_indexes(const vector<float> &v) {

  // initialize original index locations
  vector<int> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values 
  stable_sort(idx.begin(), idx.end(),
       [&v](int i1, int i2) {return v[i1] < v[i2];});

  return idx;
}



void get_comparison_frames(int start_index, int end_index, int & ref_index, int & comp_index){
    //for(int i=start_index; i<=end_index; i++){
    //    cv::Mat raw_data(std_ml_image_proc_deque.at(i)->height,std_ml_image_proc_deque.at(i)->width,CV_8UC3,std_ml_image_proc_deque.at(i)->img,cv::Mat::AUTO_STEP);
    //}
    /*int height1 = std_ml_image_proc_deque.at(start_index)->height;
    int height2 = std_ml_image_proc_deque.at(end_index)->height;
    int width1 = std_ml_image_proc_deque.at(start_index)->width;
    int width2 = std_ml_image_proc_deque.at(end_index)->width;
    char * img1 = (char *) std_ml_image_proc_deque.at(start_index)->img;
    char * img2 = (char *) std_ml_image_proc_deque.at(end_index)->img;*/

    /*
       cv::Mat raw_data(1088,1920,CV_8UC3,img,cv::Mat::AUTO_STEP);
       cout << " Got raw data " << raw_data.cols << " rows " << raw_data.rows << endl;
    */

    /*cv::Mat ref_frame2(height1,width1,CV_8UC3,img1,cv::Mat::AUTO_STEP);
    cv::Mat comp_frame2(height2,width2,CV_8UC3,img2,cv::Mat::AUTO_STEP);
    ref_index = start_index;
    comp_index = end_index;
    ref_frame = ref_frame2;
    comp_frame = comp_frame2;*/
    for(int i=start_index+1; i<end_index; i++){
        if(std_ml_image_proc_deque.at(i)->motion_flag){
          ref_index = start_index;
          comp_index = i;
          return;
        }
    }
    
    ref_index = start_index;
    comp_index = end_index;

}

cv::Mat get_mat_from_queue(int index){

    int height = std_ml_image_proc_deque.at(index)->height;
    int width = std_ml_image_proc_deque.at(index)->width;
    int filesize = std_ml_image_proc_deque.at(index)->img_size;
    char * img = (char *) std_ml_image_proc_deque.at(index)->img;
    //cout << " Size of image is " << std_ml_image_proc_deque.at(index)->img_size << endl;
    //char * img = (char *) std_ml_image_proc_deque.at(index)->img;

    /*
       cv::Mat raw_data(1088,1920,CV_8UC3,img,cv::Mat::AUTO_STEP);
       cout << " Got raw data " << raw_data.cols << " rows " << raw_data.rows << endl;
    */

    //cv::Mat* raw_data = new cv::Mat(height,width,CV_8UC3,(char *) std_ml_image_proc_deque.at(index)->img,cv::Mat::AUTO_STEP);
    //cout << " Got raw data " << raw_data->cols << " rows " << raw_data->rows << endl;
    //return raw_data;
    cv::Mat mat_img;

    mat_img = cv::imdecode(cv::Mat(1, filesize, CV_8UC1, img), IMREAD_UNCHANGED);
    return mat_img;
}

void normalize(Pixel &pixel){
    pixel.x = (pixel.x / 255.0);
    pixel.y = (pixel.y / 255.0);
    pixel.z = (pixel.z / 255.0);
}

auto mat_process(cv::Mat src, uint width, uint height) -> cv::Mat{
    // convert to float; BGR -> RGB
    cv::Mat dst;
    //cout << "Creating dst" << endl;
    src.convertTo(dst, CV_32FC3);
    //cout << "Creating dst2" << endl;
    cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);
    //cout << "Creating dst3" << endl;

    // normalize to -1 & 1
    Pixel* pixel = dst.ptr<Pixel>(0,0);
    //cout << "Creating dst4" << endl;
    const Pixel* endPixel = pixel + dst.cols * dst.rows;
    //cout << "Creating dst5" << endl;
    for (; pixel != endPixel; pixel++)
        normalize(*pixel);

    // resize image as model input
    //cout << "Creating dst6" << endl;
    cv::resize(dst, dst, cv::Size(width, height));
    //cout << "Creating dst7" << endl;
    return dst;
}

cv::Mat detect_frame(string infile, string outfile, vector<Rect>& nmsrec, vector<int>& pick,  vector<int>& ids, cv::Mat *inp){     
    // create model
    std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile((std_ml_models_path + "yolov5.tflite").c_str());
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);
    interpreter->AllocateTensors();

    // get input & output layer
    TfLiteTensor* input_tensor = interpreter->tensor(interpreter->inputs()[0]);
    TfLiteTensor* output_box = interpreter->tensor(interpreter->outputs()[0]);
    TfLiteTensor* output_score = interpreter->tensor(interpreter->outputs()[1]);

    const uint HEIGHT = input_tensor->dims->data[1];
    const uint WIDTH = input_tensor->dims->data[2];
    const uint CHANNEL = input_tensor->dims->data[3];


      cv::Mat img;
      if(inp == NULL){
        //cout << "Getting image from file " << endl;
        img = cv::imread(infile);
      }else{
        //cout << "Getting image from input " << endl;
        img = *inp; 
      }

      cv::Mat inputImg = mat_process(img, WIDTH, HEIGHT); //format the Mat for RGB from BGR

      float* inputImg_ptr = inputImg.ptr<float>(0);  //set the input
      memcpy(input_tensor->data.f, inputImg.ptr<float>(0),
            WIDTH * HEIGHT * CHANNEL * sizeof(float));


      interpreter->Invoke();


      float* output1 = interpreter->typed_output_tensor<float>(0); //get the output


      vector<cv::Rect> rects;
      vector<vector<float>> recvec;
      vector<float> scores;
      //vector<int> ids;
      vector<int> nms;
      map<int,vector<vector<float>>> map_rects;
      map<int,vector<float>> map_scores;


      

      float max_class_score = 0;
      int max_class = -1;
      float MAX_ACCEPT_SCORE = 0.20;
      float bbox_score = 0;

      int nelem = 1;
      for(int i=0; i<output_box->dims->size; ++i){

          nelem *= output_box->dims->data[i];
      }

      for(int oi=0; oi<nelem; oi++){

        if(oi % 85 == 0){ //yolov5 has 80 classes + 5 values for the bounding box, therefore each box has 85 values, if the current index goes into 85, we should process the last box

          if(max_class_score > MAX_ACCEPT_SCORE){ //if there was a max_value with class score greater than the accept score

            const float cx = output1[oi-85]; //center x of the box
            const float cy = output1[oi-84]; //center y of the box
            const float w = output1[oi-83];  //width of the box
            const float h = output1[oi-82];  //height of the box
            
            const float xmin = ((cx-(w/2))) * img.cols; //get the xmin, ymin and xmax, ymax are required by the nms used below
            const float ymin = ((cy-(h/2))) * img.rows;
            const float xmax = ((cx+(w/2))) * img.cols;
            const float ymax = ((cy+(h/2))) * img.rows;
            rects.emplace_back(cv::Rect(xmin, ymin, xmax-xmin, ymax-ymin)); //put this box in our list of rectangles
            vector<float> currvec = {xmin,ymin,xmax,ymax}; //also store the rectangle as an array as needed by the nms
            recvec.emplace_back(currvec); //put the array into the currvec
            scores.emplace_back(max_class_score); //put the max_class_score into the scores
            //cout << "Max class is " << max_class << endl;

            if (map_rects.find(max_class) == map_rects.end()) { //if the class for this box doesn't exist in the map add the class 
              map_rects[max_class] = vector<vector<float>>();
              map_rects[max_class].emplace_back(currvec);
              map_scores[max_class] = vector<float>();
              map_scores[max_class].emplace_back(max_class_score);
              // not found
            } else { //if the class for this box exists in the map, append to the map using the class as the key
              // found
              map_rects[max_class].emplace_back(currvec);
              map_scores[max_class].emplace_back(max_class_score);

            }

          }else{
            //cout << "Max class is " << max_class << " with score " << max_class_score << endl;

          }

          max_class_score = 0; //reset the class score
          max_class = -1; //reset the max class
          bbox_score = output1[oi+4]; //get the bbox score
          oi += 5; //increment 5 to be at the first class 


        }

        float test_score = output1[oi] * bbox_score; //the test score of the current index multiplied by the bbox score to get the aggregate value
        if(test_score > max_class_score){ //if it's greater than the max then use that score and also get the max_class
          max_class_score  = test_score;
          max_class = (oi % 85) - 5;
        }
      }


    //process the nms to get rid of extra boxes, but it must be processed class wise to prevent merging of occluded or over lapping classes.
    //get each set of values and then append them into a single master list of boxes and classes
    int prev_count = 0;
    for (auto element : map_rects){
      vector<cv::Rect> temp_nms_rec;
      vector<int> temp_pick;
      temp_nms_rec = nms_vec(element.second,0.5, temp_pick); 

      copy(temp_nms_rec.begin(), temp_nms_rec.end(), back_inserter(nmsrec));
      for(int index=0; index<temp_pick.size(); index++)
	    {
        pick.emplace_back(index+prev_count);
	    }
      ids.insert(ids.end(),temp_nms_rec.size(),element.first);
      prev_count += temp_nms_rec.size();

    }


    //get the pick index
    /*int pick_index = 0;
    for(cv::Rect rect: nmsrec){
        cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 3);
        cv::putText(img, to_string(ids[pick[pick_index]]), cv::Point(rect.x + 0.5 * rect.width, rect.y + 0.5 * rect.height), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(118,185,0),2);
        pick_index++; 
    }

    cv::imwrite(outfile, img);*/


    return inputImg ;
}


bool check_motion_setting(int motion_class){
  string setting = get_setting("DetectionSettings", to_string(motion_class+1), false); //add 1 because the motion classes are offset by 1 from coco
  #ifdef VIPER_DEBUG_ENABLED
    std::cout << " The request class " << to_string(motion_class) << " motion setting was " << setting << endl;
  #endif
  try{
    int val = stoi(setting);
    if(val == 1){ return true;}
    else{ return false;}

  }catch(exception &e){
    return false;
  }

}

/*bool check_zone_motion_setting(int motion_class, int zone){
  string zone = "Z"+to_string(zone);
  if(ZONES.at(zone-1).find(motion_class) == ZONES.at(zone-1).end()){
    return false;
  }else{
    return true;
  }

}*/

string motion_detect(cv::Mat mat1, cv::Mat mat2, vector<Rect> nmsrec, vector<int> pick, vector<int> ids, vector<int>& motion, bool& is_motion){
  
  
  std::chrono::time_point<std::chrono::system_clock> beg, start, end, done, nmsdone;
  std::chrono::duration<double> elapsed_seconds;
  start = std::chrono::system_clock::now();



  cv::Mat dst,cln;
  //cout << " Starting abs diff " << endl;
  absdiff(mat1,mat2, dst);
  end = std::chrono::system_clock::now();
  elapsed_seconds = start - end;
  //printf("The time for abs diff was s: %.10f\n" ,elapsed_seconds.count());

  cln = dst.clone();

  start = std::chrono::system_clock::now();
  //cout << " Got image diff " << endl;

  vector<double> eval;
  cv::Scalar eval_mean, eval_stdev;
  cv::Scalar mat_mean, mat_stdev;
  int pick_index = 0;
  double cum_area=0;
  double total_mean=0;
  double total_stdev=0;
  int channels = dst.channels();
  //cout << " Iterating over rec with channels " << channels << endl;

  //mat_mean = cv::sum(dst);
  //meanStdDev(dst,eval_mean,eval_stdev);
  //cout << " THE MAT_MEAN WAS................................." << mat_mean[0] << endl;
  
  for(cv::Rect rect: nmsrec){
    float x = rect.x;//rect.x/1920.0*320.0;
    float y = rect.y;//rect.y/1080.0*320.0;
    float width = rect.width;//rect.width/1920.0*320.0;
    float height = rect.height;//rect.height/1080.0*320.0;
    //cout << "x is " << x << " Y is " << y << " Width is " << width << " Height is " << height << endl; 

    //adjust the coordinates to fit into the image
    if(x<0){
      x = 0;
    }

    if(y<0){
      y = 0;
    }

    if(x+width > dst.cols){
      width = dst.cols-x-1;
      //cout << " Width is now " << width << endl; 
      
    }
    if(y+height > dst.rows){
      height = dst.rows-y-1;
      //cout << " Height is now " << height << endl; 
    }
    //end adjustments


    cv::Rect mod_rect = Rect(x, y, width, height);
    //cout << " Curr rect is x,y,width,height " << rect.x << " " << rect.y << " " << " " << rect.width << " " << rect.height << endl;
    //cout << " Curr mat is cols, rows" << dst.cols << " , " << dst.rows << endl; 
    Mat roi = dst(mod_rect);
    cv::meanStdDev(roi,eval_mean,eval_stdev);
    //eval_mean = cv::sum(roi);
    double sum = 0;
    double sample_stdev = 0;
    for(int i=0; i<channels; i++){
      sum += eval_mean[i];
      //cout << "sum " << sum << " Eval Mean i " << eval_mean[i] << " pick index " << pick_index << endl;
    }

    /*for(int i=0; i<channels; i++){
      sample_stdev += ((eval_mean[i] - sum/channels) * (eval_mean[i] - sum/channels))/channels; 
      cout << "sample stddev cumulative " << sample_stdev << " pick index " << pick_index << endl;
    }

    sample_stdev = sample_stdev / (mod_rect.width * mod_rect.height);
    sample_stdev = sqrt(sample_stdev);
    cout << " total std dev " << sample_stdev << " for rect " << pick_index << endl;*/
    
    eval.push_back(sum); //divide the sum of means by the number of channels to get the channel mean
    
    roi.setTo(0); //zero out the matrix
    pick_index++; 
  }

  cv::meanStdDev(dst,mat_mean,mat_stdev);
  //mat_mean = cv::sum(dst);
  //cout << " THE MAT_MEAN2 WAS..............................." << mat_mean[0] << endl;
  for(int i=0; i<channels; i++){
    total_mean += mat_mean[i];
    total_stdev += mat_stdev[i];
    //cout << " Total mat mean " << mat_mean[i] << endl;
    //cout << " Total mean " << total_mean << " std dev " << total_stdev << endl;
  }

  /*for(int i=0; i<channels; i++){
      total_stdev  += ((mat_mean[i] - total_mean/channels) * (eval_mean[i] - total_mean/channels))/channels; 
      cout << "sample stddev cumulative " << total_stdev  << " pick index " << pick_index << endl;
  }

  total_stdev = total_stdev / (dst.cols * dst.rows);
  cout << "sample stddev cumulative after col row division " << total_stdev  << endl;
  total_stdev = sqrt(total_stdev);*/
  double motion_cutoff = total_mean + 1.725 * total_stdev;
  double MOTION_CUTOFF = motion_cutoff; //(IMAGE_LEARNING_RATE * motion_cutoff) + ((1-IMAGE_LEARNING_RATE)*MOTION_CUTOFF);
  //cout << " The motion cutoff is " << motion_cutoff << " total_stddev is " << total_stdev << endl;


  end = std::chrono::system_clock::now();
  elapsed_seconds += start - end;

  //printf("The time for motion diff was s: %.10f\n" ,elapsed_seconds.count());

  //"<rt7:Object>[0,1,1,2,1,3,1,2]</rt7:Object>"
  //"<rt7:BndBox>[[0.0,0.0,0.05,0.05],[0.05,0.05,0.10,0.10],[0.10,0.10,0.15,0.15],[0.15,0.15,0.20,0.20],[0.20,0.20,0.25,0.25],[0.25,0.25,0.30,0.30],[0.30,0.30,0.35,0.35],[0.35,0.35,0.40,0.40]]</rt7:BndBox>"
  //"<rt7:ObjectZone>[[Z1,Z2],[],[Z3],[Z4],[],[],[],[]</rt7::ObjectZone>  
  //"<rt7:ZoneClasses>[[Z1:1,5,8],[Z2:21,6,12]]</rt7:ZoneClasses>"
  //"<rt7:ZoneDefinition>[[Z1, X:1,2,3, Y:2,2,2]]</rt7:ZoneDefinition>"
  string objects = "";
  string bndbox = ""; 
  string bndzones = "";
  string zoneclassstr = "";
  map<string,map<int,int>> zone_classes;
  //string zonedef="";

  pick_index = 0;
  is_motion = false;
  for(cv::Rect rect: nmsrec){
    //cout << " Testing rect as pick index " << pick_index << " which has class id " << ids[pick[pick_index]] << endl;
    //std::cout << " Cut off value is " << to_string(eval[pick_index] ) << " and motion cut off is " << to_string(MOTION_CUTOFF) << " which has class id " << ids[pick[pick_index]] << endl;
    if(eval[pick_index] > MOTION_CUTOFF){
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
      //cout << " Rect as pick index " << pick_index << " shows motion with class id " << ids[pick[pick_index]] << endl;
      cv::rectangle(cln, rect, cv::Scalar(0, 255, 0), 3);
      cv::putText(cln, to_string(ids[pick[pick_index]]), cv::Point(rect.x+0.5 * rect.width, rect.y+0.5*rect.height), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(118,185,0),2);
      objects += to_string(ids[pick[pick_index]]) + ",";
      bndbox += "[" + to_string(rect.x) + "," + to_string(rect.y) + "," + to_string(rect.x+rect.width) + "," + to_string(rect.y + rect.height) + "],"; 

      vector<string> zone_result = check_zones(rect);
      bndzones += "[";
      for(int test_res=0; test_res<zone_result.size(); test_res++){
          bndzones += zone_result.at(test_res) + ",";
          if(zone_classes.find(zone_result.at(test_res)) == zone_classes.end()){
            map<int,int> mp;
            mp[ids[pick[pick_index]]]=1;
            #ifdef VIPER_DEBUG_ENABLED
              std::cout << " Set the zone class " << ids[pick[pick_index]] << " in the zone " << zone_result.at(test_res) << endl;
            #endif
            zone_classes[zone_result.at(test_res)] = mp;
          }else{
            zone_classes[zone_result.at(test_res)][ids[pick[pick_index]]]=1;
          }
      }
      bndzones = bndzones.substr(0,bndzones.length()-1);
      bndzones += "],";


    }
    pick_index++;
  }

  for(const auto &zone_pair : zone_classes){
    map<int,int> zone_class_map = zone_pair.second;
    string zone_name = zone_pair.first;
    string zone_val = "[" + zone_name + ":";
    for(const auto &map_pair : zone_class_map){
      #ifdef VIPER_DEBUG_ENABLED
        std::cout << " The zone map has " << to_string(map_pair.first) << endl;
      #endif
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
  #ifdef VIPER_DEBUG_ENABLED
    std::cout << " Computed abs diff " << endl;
    
    //imwrite(image_name,mat2);
    std::cout << " Wrote diff image " << endl;
  #endif
  /*
    opencv absdiff get max change
    compare the max frame
    check absdiff of objects in max frame
    determine motion
    develop motion metadata
  */

  return objects + bndbox + bndzones + zoneclassstr;

}

bool clean_up_events(){

  lock_guard<mutex> image_guard(std_ml_image_event_mutex);
  #ifdef VIPER_DEBUG_ENABLED
    std::cout << " Cleaning up events size is : " << to_string(std_ml_image_event_deque.size()) << " max size is " << to_string(std_video_max_events) << endl;
  #endif
  if(std_ml_image_event_deque.size() > std_video_max_events){
      int num_del = std_ml_image_event_deque.size() - std_video_max_events;
      for(int i=0; i<num_del; i++){
        std_ml_image_event_deque.pop_front(); //remove the element at 0 or the oldest element
      }
      std_ml_image_event_deque_offset += num_del; //increase the offset by numdel so that when we request an index it will account for this
      std_ml_image_event_file_num_del += num_del; //add to the number of deletions in the event list
  }
  #ifdef VIPER_DEBUG_ENABLED
    std::cout << " Cleaning up events size now is : " << to_string(std_ml_image_event_deque.size()) << " max size is " << to_string(std_video_max_events) << endl;
  #endif
  return true;

}

string get_meta_data(string meta_path){
  string meta_data = "";
  string line = "";
  ifstream metafile(meta_path);
  if(metafile.is_open()){
    while(getline(metafile,line))
    {
      meta_data += line;
    }
      metafile.close();
  }
  return meta_data;
}


void get_image_events_seq(int refseq, int num, bool seek_newest, vector<string>& seqlist){

  //if(num > std_video_max_request_events){
  //   num = std_video_max_request_events;  //maximum 10 images at a time
  //}

  lock_guard<mutex> image_guard(std_ml_image_event_mutex);
  bool add_curr_img = false;

  string latest_cam_image = "";

  //if reqseq is negative the latest image is put at the top of the list
  int indexoffset = -1;
  if(refseq == -1){
    if(std_ml_image_event_deque.size() > 0){
      refseq = std_ml_image_event_deque.at(std_ml_image_event_deque.size()-1).seq; //get the latest ref seq
      indexoffset = 0;
    }

  }
  
  if(std_ml_image_event_index_map.find(refseq)== std_ml_image_event_index_map.end()){
    //refseq not found
    #ifdef VIPER_DEBUG_ENABLED
      std::cout << " No events..." << endl;
    #endif
    
    #ifdef VIPER_DEBUG_ENABLED
      std::cout << " Adding the latest frame to the response..." << endl;
    #endif
    
    int seq = get_std_video_max_seq()-2;
    seqlist.push_back(to_string(seq));

    return;
    
  }else{
      //ref seq found
      int index = std_ml_image_event_index_map[refseq]-std_ml_image_event_deque_offset;
      #ifdef VIPER_DEBUG_ENABLED
        std::cout << " The index is " << to_string(index) << " off set is " << to_string(std_ml_image_event_deque_offset) << endl;
      #endif
      //the -1 seq is added first to the list
      

      
      int dir = -1;
      int start_index, end_index;
      if(seek_newest){ 
        // index --> newer frames (index++) index is the youngest frame
        start_index = index + num - indexoffset;
        if(start_index > (std_ml_image_event_deque.size()-1)){
          start_index = std_ml_image_event_deque.size()-1;
          end_index = start_index - num;
        }
        else{end_index = index;}
      }
      else{    
        // older frames  <--- index (index--) index is the oldest frame
        start_index = index + indexoffset;
        end_index = index+indexoffset-num;
        if(end_index < 0){
          end_index = 0;
        } 

      }


      for(int i=0; i<num;  i++){
        #ifdef VIPER_DEBUG_ENABLED
          cout << " The start index is " << to_string(start_index) << endl;
        #endif
        if(start_index >= std_ml_image_event_deque.size() || start_index < 0){ break;}
        viper_sequence curr_seq = std_ml_image_event_deque.at(start_index);
        seqlist.push_back(to_string(curr_seq.seq)); //---------------------> add the seq list
        start_index += dir;
        #ifdef VIPER_DEBUG_ENABLED
          cout << " The start index is " << to_string(start_index) << " seq is " <<  to_string(curr_seq.seq) << endl;
        #endif
      }
  }



}

void get_image_events(int refseq, int num, bool seek_newest, vector<string>& seqlist, vector<string>& images, vector<string>& meta, vector<string>& frameseq, vector<string>& image_names){

  if(num > std_video_max_request_events){
     num = std_video_max_request_events;  //maximum 10 images at a time
  }

  lock_guard<mutex> image_guard(std_ml_image_event_mutex);
  bool add_curr_img = false;

  string latest_cam_image = "";

  //if reqseq is negative the latest image is put at the top of the list
  int indexoffset = -1;
  if(refseq == -1){
    if(std_ml_image_event_deque.size() > 0){
      refseq = std_ml_image_event_deque.at(std_ml_image_event_deque.size()-1).seq; //get the latest ref seq
      indexoffset = 0;
    }

    
    string img_path = std_videos_save_dir + std_videos_image_file_name; //file location
    #ifdef VIPER_DEBUG_ENABLED
      std::cout << " Trying to get the image " << img_path << endl;
    #endif
    lock_guard<mutex> video_guard(std_videos_img_mutex); //locak the camera image
      std::ifstream input( img_path , std::ios::binary ); //read the file as binary
      // copies all data into buffer
      std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(input), {});
      #ifdef VIPER_DEBUG_ENABLED
        std::cout<<"The size of the vector is "<<to_string(buffer.size())<<endl;
      #endif
      //convert buffer into base64 array
      unsigned char* buff = &buffer[0];
      //encode as base64 image
      //string b64_cam_image1 = base64_encode_str(buff, buffer.size());  
      latest_cam_image = base64_encode(buff, buffer.size(),false);  
      input.close();
    //std::cout << " Got the current image file and released the lock " << latest_cam_image << endl;
  }
  
  if(std_ml_image_event_index_map.find(refseq)== std_ml_image_event_index_map.end()){
    //refseq not found
    #ifdef VIPER_DEBUG_ENABLED
      std::cout << " No events..." << endl;
    #endif
    if(latest_cam_image != ""){
      #ifdef VIPER_DEBUG_ENABLED
        std::cout << " Adding the latest frame to the response..." << endl;
      #endif
      int seq = get_std_video_max_seq()-2;
      seqlist.push_back(to_string(seq));
      images.push_back(latest_cam_image);
      image_names.push_back(std_videos_image_file_name);

      string meta_path = std_videos_save_dir + to_string(seq) + std_video_metadata_extension;
      string meta_data = get_meta_data(meta_path);
      meta.push_back(meta_data);
      frameseq.push_back("");
      return;
    }

  }else{
      //ref seq found
      int index = std_ml_image_event_index_map[refseq]-std_ml_image_event_deque_offset;
      #ifdef VIPER_DEBUG_ENABLED
        std::cout << " The index is " << to_string(index) << " off set is " << to_string(std_ml_image_event_deque_offset) << endl;
      #endif
      //the -1 seq is added first to the list
      
      if(latest_cam_image != ""){
            #ifdef VIPER_DEBUG_ENABLED
              std::cout << " Seeking older with refseq as -1 means there is nothing to return " << endl;
              std::cout << " Adding the latest frame to the response..." << endl;
            #endif
            int seq = get_std_video_max_seq()-2;
            seqlist.push_back(to_string(seq));
            images.push_back(latest_cam_image);
            image_names.push_back(std_videos_image_file_name);

            string meta_path = std_videos_save_dir + to_string(seq) + std_video_metadata_extension;
            string meta_data = get_meta_data(meta_path);
            meta.push_back(meta_data);
            frameseq.push_back("");
      } //don't return anything if seq is specified
      


      int dir = -1;
      int start_index, end_index;
      if(seek_newest){ 
        // index --> newer frames (index++) index is the youngest frame
        start_index = index + num - indexoffset;
        if(start_index > (std_ml_image_event_deque.size()-1)){
          start_index = std_ml_image_event_deque.size()-1;
          end_index = start_index - num;
        }
        else{end_index = index;}
      }
      else{    
        // older frames  <--- index (index--) index is the oldest frame
        start_index = index + indexoffset;
        end_index = index+indexoffset-num;
        if(end_index < 0){
          end_index = 0;
        } 

      }


      for(int i=0; i<num;  i++){
        #ifdef VIPER_DEBUG_ENABLED
          cout << " The start index is " << to_string(start_index) << endl;
        #endif
        if(start_index >= std_ml_image_event_deque.size() || start_index < 0){ break;}
        viper_sequence curr_seq = std_ml_image_event_deque.at(start_index);
        seqlist.push_back(to_string(curr_seq.seq)); //---------------------> add the seq list
        string img_path = std_videos_save_dir + to_string(curr_seq.seq) + "-" + curr_seq.image_path + std_video_image_extension;

        std::ifstream input( img_path , std::ios::binary ); //read the file as binary
        // copies all data into buffer
        std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(input), {});
        input.close();
        //cout<<"The size of the vector is "<<to_string(buffer.size())<<endl;
        //convert buffer into base64 array
        unsigned char* buff = &buffer[0];
        //encode as base64 image
        string b64_cam_image = base64_encode(buff, buffer.size(),false);
        images.push_back(b64_cam_image); //---------------------> add the image
        image_names.push_back(to_string(curr_seq.seq) + "-" + curr_seq.image_path + std_video_image_extension); //--------------> add the image name

        string meta_path = std_videos_save_dir + to_string(curr_seq.seq) + std_video_metadata_extension;
        string meta_data = get_meta_data(meta_path);
        
        meta.push_back(meta_data); //---------------------> add the meta data 
        frameseq.push_back(to_string(curr_seq.frameseq));//----------------------------->add frameseq
        start_index += dir;
        #ifdef VIPER_DEBUG_ENABLED
          cout << " The start index is " << to_string(start_index) << endl;
        #endif
      }

      //if we are going from 
      //if(latest_cam_image != "" && dir == -1){
      //  seqlist.push_back(to_string(-1));
      //  images.push_back(latest_cam_image);
      //  meta.push_back("");
      //}
  }


}

bool save_events_to_file(){
  try{
    lock_guard<mutex> image_guard(std_ml_image_event_mutex);
    //std_ml_image_event_file_last_index
    //std_ml_image_event_file_name
    string events = "";
    int count = 0;
    for(int i=std_ml_image_event_file_last_index-std_ml_image_event_deque_offset; i<std_ml_image_event_deque.size();i++){
        events += to_string(std_ml_image_event_deque.at(i).seq) + " " + to_string(std_ml_image_event_deque.at(i).frameseq) + " " + std_ml_image_event_deque.at(i).localtime_str + "\n";
        count++;
    }
    #ifdef VIPER_DEBUG_ENABLED
      cout << " Saved events " << events << endl; 
    #endif

    if(events.length() > 0){
      std::ofstream output(std_ml_image_event_file_name,std::ios_base::app);
      output << events;
      std_ml_image_event_file_last_index += count;
    }

    #ifdef VIPER_DEBUG_ENABLED
      std::cout << " Saved events to file " << endl;
    #endif

    return true;
  }catch(exception &e){
    std::cout << " Exception in save events .... " << endl;
    return false;
  }
   
}

bool delete_events_from_file(){
  try{
    lock_guard<mutex> image_guard(std_ml_image_event_mutex);
    int num_events = std_ml_image_event_file_num_del;
    if(num_events==0) { return true;}
    //std_ml_image_event_file_last_index
    //std_ml_image_event_file_name
    std::ifstream input(std_ml_image_event_file_name);
    std::ofstream output(std_ml_image_event_file_name+".tmp");

    int i=0;
    if(input.good()){
      std::string line;
      while(std::getline(input,line)){
        i+=1;
        if(i<=num_events){ continue;}
        output << line << std::endl;
      }
    }
    input.close();
    output.close();
    rename((std_ml_image_event_file_name+".tmp").c_str(), std_ml_image_event_file_name.c_str());
    std_ml_image_event_file_num_del = 0;
    //std_ml_image_event_file_last_index = map_index; //*******************
    #ifdef VIPER_DEBUG_ENABLED
      std::cout << " Deleted events from file " << endl;
    #endif

    return true;
  }catch(exception &e){
    std::cout << " Exception in delete events .... " << endl;
    return false;
  }

}

bool load_events_from_file(){
  try{
    lock_guard<mutex> image_guard(std_ml_image_event_mutex);
    //std_ml_image_event_file_last_index
    //std_ml_image_event_file_name
    //segment_id  frameseq map_index
    std::ifstream input(std_ml_image_event_file_name);
    int segment_id; int meta_index;
    string segment_id_s; string meta_index_s; string localtime_str; string time_str;
    int i=1;
    std_ml_image_event_deque_offset = 0; //offset is reset to 0 on load
    if(input.good()){
      std::string line;
      while(std::getline(input,line)){
        std::istringstream iss(line);
        if(!(iss >> segment_id_s >> meta_index_s >> localtime_str >> time_str)){
          break;
        }
        segment_id = stoi(segment_id_s);
        meta_index = stoi(meta_index_s);
        

        viper_sequence vip_seq = viper_sequence(segment_id,to_string(meta_index),meta_index, localtime_str + " " + time_str);
        std_ml_image_event_deque.push_back(vip_seq); //push the event onto the queue
        std_ml_image_event_index_map[segment_id] = i-1; //set the index in the list
        std_ml_image_event_file_last_index = i; //the last index is plus 1
        cout << " Loaded event " << to_string(segment_id) <<" "<< to_string(meta_index) <<" "<< endl; 
        i++;
              
      }
      
      
    }
    std_ml_image_event_file_num_del = 0;
    input.close();
    #ifdef VIPER_DEBUG_ENABLED
      std::cout << " Loaded events from file ...  " << endl;
    #endif
    return true;
  }catch(exception &e){
    std::cout << " Exception in load events .... " << endl;
    return false;
  }

}

string process_frame(int & segment_id, int & index, int & max_index, int offset, int PROCESS_RATE, bool& first_frame){

    cv::Mat ref_frame;
    cv::Mat comp_frame;
    try{

        


        viper_image * latest_frame = std_ml_image_proc_deque.front();
        
        int curr_segment_id = segment_id = latest_frame->segment_id;
        index = 0;
        max_index=0;
        //int currindex = index;

        #ifdef VIPER_DEBUG_ENABLED
          std::cout << "Got the index and max index and segment id " << index << " " << max_index << " " << segment_id << endl;
        #endif
        int ref_index=0;
        //int comp_index=0;

        ref_frame = get_mat_from_queue(ref_index);
        #ifdef VIPER_DEBUG_ENABLED
          std::cout << "GOT THE MAT FROM THE QUEUE " << index << " " << max_index << " " << segment_id << endl;
        #endif
        lock_guard<mutex> avg_guard(avg_mat_mutex); //lock the avg mat mutex
        comp_frame = imgv_mat_avg;//get_mat_from_queue(comp_index);


        vector<Rect> nmsrec1;//, nmsrec2; 
        vector<int> pick1;//, pick2;  
        vector<int> ids1;//, ids2;
        vector<int> motion1;//;, motion2;
        cv::Mat pmat1 = detect_frame("", "./result0.jpg", nmsrec1, pick1, ids1, &ref_frame);



        #ifdef VIPER_DEBUG_ENABLED
          std::cout << " Starting motion detect " << endl;
        #endif
        string meta_result = "";
        
        int meta_index = 0;
        cv::Mat meta_frame;
        bool is_motion=false;

        //use comp index
        meta_index = ref_index;
        meta_frame = ref_frame;
        meta_result = motion_detect(comp_frame, ref_frame, nmsrec1, pick1, ids1, motion1, is_motion); 

        if(motion1.size() > 0){
          //there is motion only update comp frame index because the object bounding boxes are extracted from comp_frame
          #ifdef VIPER_DEBUG_ENABLED
            std::cout << "UPDATING MOTION REF " << endl;
          #endif
          update_rl_controller(std_ml_image_proc_deque.at(ref_index)->rl_index, std_ml_image_proc_deque.at(ref_index)->motion_flag, true);
        }else{
          //there is no motion only update comp frame index because the object bounding boxes are extracted from comp_frame 
          #ifdef VIPER_DEBUG_ENABLED
            std::cout << "UPDATING NO MOTION REF " << endl;
          #endif
          update_rl_controller(std_ml_image_proc_deque.at(ref_index)->rl_index, std_ml_image_proc_deque.at(ref_index)->motion_flag, false);
        }         
        

        string output = "";
        if(is_motion){
            #ifdef VIPER_DEBUG_ENABLED
              std::cout << " FOUND MOTION PUSHING SEQ INTO MOTION QUEUE OF SIZE " << to_string(std_ml_image_event_deque.size()) << endl;
            #endif
            std::tm localtm = latest_frame->localtime;
            string localtm_str = datetm_to_str(localtm);
            lock_guard<mutex> image_guard(std_ml_image_event_mutex);
            viper_sequence vip_seq = viper_sequence(segment_id,to_string(meta_index+offset),meta_index+offset, localtm_str);
            if((std_ml_image_event_deque.size() > 0 && std_ml_image_event_deque.back().seq != segment_id) || (std_ml_image_event_deque.size() == 0)){ //don't push the event if it already is there for the current segment
              std_ml_image_event_deque.push_back(vip_seq);
              std_ml_image_event_index_map[segment_id] = std_ml_image_event_deque.size() - 1 + std_ml_image_event_deque_offset;
            }

            string image_name = /*std_videos_save_dir +*/ to_string(segment_id) + "-" + to_string(meta_index+offset) + std_video_image_extension;
            string image_ref = "<rt7:Jpeg>" + image_name + "</rt7:Jpeg>";
            cv::imwrite(std_videos_save_dir +image_name, meta_frame, {cv::IMWRITE_JPEG_QUALITY,std_videos_jpeg_quality});
            output = "<rt7:Frame>"
            "<rt7:FrameSeq>" + to_string(meta_index+offset) + "</rt7:FrameSeq>" +
            "<rt7:FrameTime>" + to_string(std_ml_image_proc_deque.at(meta_index)->frame_time)+"</rt7:FrameTime>"
            + meta_result + image_ref + 
            "</rt7:Frame>";
        }/*else if(!first_frame){
            string image_name = to_string(segment_id) + "-" + to_string(meta_index+offset) + std_video_image_extension;
            string image_ref = "<rt7:Jpeg>" + image_name + "</rt7:Jpeg>";
            cv::imwrite(std_videos_save_dir +image_name, meta_frame);
            output = "<rt7:Frame>"
            "<rt7:FrameSeq>" + to_string(meta_index+offset) + "</rt7:FrameSeq>" +
            "<rt7:FrameTime>" + to_string(std_ml_image_proc_deque.at(meta_index)->frame_time - std_ml_image_proc_deque.at(meta_index)->segment_time) +"</rt7:FrameTime>"
            + meta_result + image_ref + 
            "</rt7:Frame>";
        }else{

        }*/



        return output;
    }
    catch (exception& e)
    {
        std::cout << "Exception in the function " << e.what() << '\n';
        std::cout << "start delete" << endl;

        /*if(ref_frame != NULL){
            delete ref_frame;
        }
        if(comp_frame != NULL){
            delete comp_frame;
        }*/

        std::cout << "done delete" << endl;
        
        return "";
    }
}

void delete_objects(int index, int max_index){
    { //when delete the image queue need to manage the lock guard
        #ifdef VIPER_DEBUG_ENABLED
          std::cout << "Starting deletion of frames index: " << index << " max index: " << max_index << " starting size " << std_ml_image_proc_deque.size() << endl;
        #endif
        lock_guard<mutex> frame_guard(frame_mutex);
        for(int i=index; i<=max_index; i++){
            viper_image * latest_frame = std_ml_image_proc_deque.front();
            std_ml_image_proc_deque.pop_front();
            //log.msg("After pop front...");
            delete [] latest_frame->img;
            delete latest_frame;
        }
        #ifdef VIPER_DEBUG_ENABLED
          std::cout << "Ending deletion of frames index: " << index << " max index: " << max_index << " starting size " << std_ml_image_proc_deque.size() << endl;
        #endif
    }
}

int ml_processing_thread(run* runner){
    debug_log log("ml_processing_thread", "main");
    try{
        int curr_segment_id = -1;
        int i = 0;
        int offset = 0;
        int segment_id=0;
        std_ml_image_event_deque_offset = 0;
        bool first_frame = false;
        check_load_zones();
        load_events_from_file();
        
  
        while (keep_ml_thread_alive()) //abstracted out for thread safety
        {
            //pop out the oldest item  pop_front();
            //cout << " Running thread " << std_ml_image_proc_deque.size() << endl;
            check_move_images_from_preprocessing(); //preprocess the images 
            ml_image_mgr_check_delete(); //delete any old images from the queue
            
            if(std_ml_image_proc_deque.size() > 0){
              #ifdef VIPER_DEBUG_ENABLED
                log.msg("Processing viper frame...at PROCESS_RATE:" + to_string(PROCESS_RATE));
              #endif
              segment_id=0;
              int index, max_index;
              string output = process_frame(segment_id, index, max_index, offset, PROCESS_RATE, first_frame);
              if(curr_segment_id == -1){
                curr_segment_id = segment_id;
              }

              offset += max_index+1;
              #ifdef VIPER_DEBUG_ENABLED
                std::cout << "Got output " << output << endl;
              #endif
              delete_objects(index, max_index);
                
                
              if(std_ml_fragment_objects_map.find(segment_id) == std_ml_fragment_objects_map.end()){
                    
                //cout << latest_frame->segment_id << " new output is: " << output << endl;
                string zonedef = "<rt7:ZoneDefinition>[" + std_ml_zone_def_str + "]</rt7:ZoneDefinition>";
                std_ml_fragment_objects_map[segment_id] = zonedef + output;

              }else{
                std_ml_fragment_objects_map[segment_id] = std_ml_fragment_objects_map[segment_id] + output;
                    //cout << latest_frame->segment_id << " appended output is: " << std_ml_fragment_objects_map[latest_frame->segment_id]  << endl;
              }
            }


            int queue_size = std_ml_image_proc_deque.size();
            int curr_file_segment = get_std_video_max_seq();
            if(queue_size == 0 && curr_segment_id != -1){
              segment_id = get_std_video_max_seq(); 
            } //queue is zero and curr segment is active advance the segmenet counter


            //log.msg( std_ml_fragment_objects_map[latest_frame->segment_id] );
            if(curr_segment_id != segment_id && curr_segment_id != -1){
              #ifdef VIPER_DEBUG_ENABLED
                log.msg("Segment has changed....");
              #endif
              //segment has changed
              string curr_segment_str = std_ml_fragment_objects_map[curr_segment_id];
              first_frame = false;
              //log.msg("Saving segment:" + to_string(curr_segment_id) + " with data" + curr_segment_str);
              //save the current segment here
              //destroy the segment from map
              string meta_file = std_videos_save_dir + to_string(curr_segment_id) + std_video_metadata_extension;
              std::ofstream out(meta_file, std::ofstream::app);
              out << curr_segment_str;
              out.close();

              std_ml_fragment_objects_map.erase(curr_segment_id);
              curr_segment_id = -1;
              offset = 0;
              #ifdef VIPER_DEBUG_ENABLED
                log.msg("Curr segment is now " + to_string(curr_segment_id) + " map size is " + to_string(std_ml_fragment_objects_map.size()));
              #endif
            }
            
            //log.msg("outside the loop in ml processing thread...");
            usleep(std_thread_ml_sleep_microseconds); 
        }

    }catch (const std::exception& e){
        std::cout << "EXCEPTION IN THE ML THREAD ---------------------------------------------" << endl;
        log.msg(e.what());
    }
    


    return 0;   

}