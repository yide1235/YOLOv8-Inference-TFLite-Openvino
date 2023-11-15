#include <vector>
#include <opencv4/opencv2/opencv.hpp>
std::vector<cv::Point3f> triangulatePoints(const cv::Mat& K_l, const cv::Mat& K_r, const cv::Mat& R, const cv::Mat& T,
                                           const std::vector<std::vector<cv::Point2f>>& uvs_l, const std::vector<std::vector<cv::Point2f>>& uvs_r);