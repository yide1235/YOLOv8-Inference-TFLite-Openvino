





#ifndef ML_ROCESSING_THREAD_HPP
#define ML_ROCESSING_THREAD_HPP

#include <string>
#include <map>
#include <vector>
#include <ctime>
#include "include/onvif/event_service/event_service.hpp"
#include "include/run.h"
#include <gst/gst.h>
#include <deque>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;
using namespace cv;

typedef cv::Point3_<float> Pixel;
class viper_image
{
  public:
    unsigned char* img; // the jpeg image
    int img_size, width, height; //the size of the char* image
    guint64 frame_time; //this is the nanosecond value of the frame subtracting frame_time - segment_time gives the time into the video
    guint64 segment_time; //this is a nanosecond value signifying the time the segment started
    int segment_id; //saving 1.mp4 2.mp4 where 1 and 2 are segment_id which increases incrementally
    std::tm localtime;
    bool motion_flag;
    int rl_index; //the index of the num standard deviations of motion for the RL controller
    viper_image(unsigned char* img, int img_size, guint64 frame_time, guint64 segment_time, std::tm localtime, int segment_id, int width, int height, bool motion_flag, int rl_index);
    ~viper_image();
};

class viper_zone
{
  public:
    vector<float> vertx;
    vector<float> verty;
    string zone_x;
    string zone_y;
    int vert;
    string zone_id;
    viper_zone(vector<float> vertx, vector<float> verty, int vert, string zone_id, string zone_x, string zone_y);
    ~viper_zone();

};

class viper_sequence
{
  public:
    int seq; //the current sequence
    string image_path; //current image path
    int frameseq; //the frame sequence
    string localtime_str; //the sequence datetime
    viper_sequence(int seq, string image_path, int frameseq, string localtime_str);
    ~viper_sequence();
};

//these are not declared in map_set to ensure that there is a persistent reference with the thread. 
static map<int,string> std_ml_fragment_objects_map; //map of the current meta files

static deque<viper_sequence> std_ml_image_event_deque; //all the viper events in order
static int std_ml_image_event_deque_offset; //as events are removed from the dequeue we offset the mapped index by this amount;
static map<int,int> std_ml_image_event_index_map; //maps <sequence->index> in the dequeue
static mutex std_ml_image_event_mutex; //mutex to protect access to the queue and allow maintenance thread to safely delete values
static int std_ml_image_event_file_last_index; //remember the last index that was saved to file
static int std_ml_image_event_file_num_del; //number of deletions since the last event clean up

static deque<viper_image*> std_ml_image_proc_deque_preprocess; //queue of the images before their sequence is confirmed
static deque<viper_image*> std_ml_image_proc_deque; //queue of the images
static deque<viper_zone> std_ml_camera_zones_deque; //queue of the zones
static bool std_ml_camera_zones_init_state=false; //decides if the zones are initialized
static string std_ml_zone_def_str; //zone defition [Z1, X:[1,2,3] , Y:[2,3,4] ], ...


//end of declared objects

int ml_processing_thread(run* runner);
void ml_load_image(unsigned char* img, int img_size, guint64 frame_time, guint64 segment_time, std::tm localtime, int segment_id , int width, int height);
int horz_intersect(float x1, float y1, float x2, float y2, float testx, float testy);
int pnpoly(int nvert, float* vertx, float* verty, float testx, float testy);
vector<float> get_floats(string s);
bool check_load_zones();
vector<string> check_zones(cv::Rect test_rec);
vector<int> sort_indexes(const vector<float> &v);
void get_comparison_frames(int start_index, int end_index, int & ref_index, int & comp_index);
cv::Mat get_mat_from_queue(int index);
void normalize(Pixel &pixel);
cv::Mat detect_frame(string infile, string outfile, vector<Rect>& nmsrec, vector<int>& pick,  vector<int>& ids, cv::Mat *inp);
string motion_detect(cv::Mat mat1, cv::Mat mat2, vector<Rect> nmsrec, vector<int> pick, vector<int> ids, vector<int>& motion, bool& is_motion);

bool clean_up_events();//clean up the event queue if the size is larger that allowed
bool save_events_to_file(); //save queued events to file
bool delete_events_from_file(); //delete events from file
bool load_events_from_file(); //load events from file

string process_frame(int & segment_id, int & index, int & max_index, int offset, int PROCESS_RATE);
void delete_objects(int index, int max_index);
long avg_array(unsigned char* img, int img_size);
bool check_motion_setting(int motion_class);
   
tuple<int,int,string> search_events_by_datetime(std::tm start_date, std::tm end_date);
void get_image_events(int refseq, int num, bool seek_newest, vector<string>& seqlist, vector<string>& images, vector<string>& meta, vector<string>& frameseq, vector<string>& image_names);
void get_image_events_seq(int refseq, int num, bool seek_newest, vector<string>& seqlist);
bool clean_up_events();
void check_move_images_from_preprocessing(); //move images from preprocessing to processing as their sequence becomes known

#endif