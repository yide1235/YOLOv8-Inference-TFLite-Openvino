#include <iostream>
#include <thread>


#include "../../include/threading/image_processing_thread.hpp"
#include "../../include/threading/yolov8_image_processing_thread.hpp"
#include "../../include/util/map_set.h"

using namespace std;


// COMPILE WITH `g++ "-I/usr/include/opencv4" $(find src -name '*.cpp') -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -o main.exe`

int main(){

}

int image_processing_thread()
{

    // START ALL THREADS

    thread thread_yolov8_image_processing(yolov8_image_processing_thread);


    // GET FRAMES FROM VIDEOS

    // RUN YOLOV8 ON THOSE FRAMES 
    std::unique_lock<std::mutex> yolov8_processing_lock(yolov8_processing_mutex);
    yolov8_processing_lock.lock();
    yolov8_processing_start_cv.notify_all();
    yolov8_processing_lock.unlock();

    // RUN YOLOV8 POST PROCESSING ON THOSE FRAMES , notified at end of yolov8 processing thread
    

    thread_yolov8_image_processing.join();
    return 0;

}