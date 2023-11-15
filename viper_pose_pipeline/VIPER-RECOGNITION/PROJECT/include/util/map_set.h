#ifndef map_set_hpp
#define map_set_hpp

#include <iostream>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <opencv4/opencv2/opencv.hpp>

using namespace std;

extern mutex yolov8_processing_mutex;
extern condition_variable yolov8_processing_start_cv;
extern mutex yolov8_post_process_mutex;
extern condition_variable yolov8_post_process_start_cv;
extern mutex person_database_management_mutex;
extern condition_variable person_database_management_start_cv;

#endif map_set_hpp