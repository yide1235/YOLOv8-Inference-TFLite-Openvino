#include "../../include/threading/yolov8_post_process_thread.hpp"
#include "../../include/util/map_set.h"
using namespace std;

std::unique_lock<std::mutex> yolov8_post_process_lock(yolov8_post_process_mutex);

bool keep_yolov8_post_process_thread_alive(){
    
    yolov8_post_process_lock.lock();
    bool keep_alive = true;
    yolov8_post_process_lock.unlock();

    return keep_alive;
}

int yolov8_post_process_thread()
{
    while(keep_yolov8_post_process_thread_alive()){
        // PULL OUT DATA FROM YOLOV8 RESULT DATA FILE
        // CHECK CLASSES AND IF A PERSON CREATE PERSON
        // INCLUDES TRIANGULATION OF POINTS, CHECKING POINTS IN SCREEN AND GETTING MEASUREMENTS
        // ADD PERSON TO FILE OF PEOPLE TO ADD TO DATABASE

        // NOTIFY NEXT THREAD OF DATA READY FOR POST PROCESSING
        yolov8_post_process_lock.lock();
        person_database_management_start_cv.notify_all();
        yolov8_post_process_lock.unlock(); 
    }
}