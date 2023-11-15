#include <vector>
#include <opencv4/opencv2/opencv.hpp>
cv::Point2i sweep_line_block(const cv::Mat& src_img, const cv::Mat& trg_img, const cv::Point2i& src_point, int height_radius, int width_radius);