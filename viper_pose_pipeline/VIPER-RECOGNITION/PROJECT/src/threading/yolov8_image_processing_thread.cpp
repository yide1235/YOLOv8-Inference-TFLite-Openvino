#include "../../include/threading/yolov8_image_processing_thread.hpp"
#include "../../include/util/map_set.h"
using namespace std;

std::unique_lock<std::mutex> yolov8_processing_lock(yolov8_processing_mutex);
std::vector<cv::Mat> next_image_to_process_with_yolov8;


bool keep_yolov8_image_processing_thread_alive(){
    
    yolov8_processing_lock.lock();
    bool keep_alive = true;
    yolov8_processing_lock.unlock();

    return keep_alive;
}

int yolov8_image_processing_thread()
{
    while(keep_yolov8_image_processing_thread_alive()){

        yolov8_processing_lock.lock();

        // READ IN FIRST IMAGE IN LIST
        cv::FileStorage file("images_to_process_with_yolov8.yml", cv::FileStorage::READ);
        if (file.isOpened()) {
            cv::FileNode imagesNode = file["images"];
            if (imagesNode.type() == cv::FileNode::SEQ) {
                for (const auto& imageNode : imagesNode) {
                    cv::Mat image;
                    imageNode["data"] >> image;

                    if (!image.empty()) {
                        next_image_to_process_with_yolov8.push_back(image);
                    }
                }
            }
        }

        file.release();

        if (!next_image_to_process_with_yolov8.empty()){
            cv::FileStorage file("images_to_process_with_yolov8.yml", cv::FileStorage::WRITE);
            file << "images" << "[";
            std::vector<cv::Mat> images_to_push_to_yolov8_to_process_file(next_image_to_process_with_yolov8.begin() + 1, next_image_to_process_with_yolov8.end());
            next_image_to_process_with_yolov8={next_image_to_process_with_yolov8[0]};
            for (const auto& image : images_to_push_to_yolov8_to_process_file) {
                file << "{:";
                file << "data" << image;
                file << "}";
            }
            file << "]";
            file.release();
        }

        yolov8_processing_lock.unlock();

        // HANDLE EMPTY LIST
        if (next_image_to_process_with_yolov8.empty()){

            yolov8_processing_start_cv.wait(yolov8_processing_lock);

            yolov8_processing_lock.lock();

            // READ IN FIRST IMAGE IN LIST
            cv::FileStorage file("images_to_process_with_yolov8.yml", cv::FileStorage::READ);
            if (file.isOpened()) {
                cv::FileNode imagesNode = file["images"];
                if (imagesNode.type() == cv::FileNode::SEQ) {
                    for (const auto& imageNode : imagesNode) {
                        cv::Mat image;
                        imageNode["data"] >> image;

                        if (!image.empty()) {
                            next_image_to_process_with_yolov8.push_back(image);
                        }
                    }
                }
            }

            file.release();

            if (!next_image_to_process_with_yolov8.empty()){
                cv::FileStorage file("images_to_process_with_yolov8.yml", cv::FileStorage::WRITE);
                file << "images" << "[";
                std::vector<cv::Mat> images_to_push_to_yolov8_to_process_file(next_image_to_process_with_yolov8.begin() + 1, next_image_to_process_with_yolov8.end());
                next_image_to_process_with_yolov8={next_image_to_process_with_yolov8[0]};
                for (const auto& image : images_to_push_to_yolov8_to_process_file) {
                    file << "{:";
                    file << "data" << image;
                    file << "}";
                }
                file << "]";
                file.release();
            }

            yolov8_processing_lock.unlock();
        }

        // RUN YOLOV8 ON ONLY IMAGE IN `next_image_to_process_with_yolov8`

        // NEED TO WRITE RESULTS TO A FILE TO LATER POST PROCESS

        // NOTIFY NEXT THREAD OF IMAGES READY FOR POST PROCESSING
        yolov8_processing_lock.lock();
        yolov8_post_process_start_cv.notify_all();
        yolov8_processing_lock.unlock(); 

    }
    return 0;
}