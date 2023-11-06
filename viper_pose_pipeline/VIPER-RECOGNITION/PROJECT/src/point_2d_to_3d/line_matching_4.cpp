#include <iostream>
#include <vector>
#include <opencv4/opencv2/opencv.hpp>
#include "../../include/point_2d_to_3d/ssim.h" 

cv::Point2i sweep_line_block(const cv::Mat& src_img, const cv::Mat& trg_img, const cv::Point2i& src_point, int height_radius, int width_radius) {
    // takes in source image (src_img), target image (trg_img), point in source image (src_point), width and height radius (half of size) for square around pixel
    // images are to be rectified
    // returns corresponding pixel location in target image
    
    int row = src_point.y;
    int src_col = src_point.x;
    cv::Mat src_block = src_img(cv::Range(row - height_radius, row + height_radius + 1),
                                 cv::Range(src_col - width_radius, src_col + width_radius + 1));
    std::vector<double> scores;
    for (int i = width_radius; i < trg_img.cols - width_radius; ++i) {
        cv::Mat trg_block = trg_img(cv::Range(row - height_radius, row + height_radius + 1),
                                     cv::Range(i - width_radius, i + width_radius + 1));
        double score_ssim = ssim(src_block, trg_block, 2, true, 7); // Assuming 'ssim' function returns a double
        scores.push_back(score_ssim);
    }
    int max_index = std::distance(scores.begin(), std::max_element(scores.begin(), scores.end()));
    return cv::Point2i(max_index + width_radius, src_point.y);
}


// int main() {
//     cv::Mat src_image = cv::imread("../../Dropbox/REALTIME7/tmpl.jpg");
//     cv::Mat trg_image = cv::imread("../../Dropbox/REALTIME7/tmpr.jpg");
//     cv::Point2i src_point(1148, 742);
//     int height_radius = 10;
//     int width_radius = 10;

//     cv::Point2i result = sweep_line_block(src_image, trg_image, src_point, height_radius, width_radius);
//     std::cout << "Result: " << result.x << ", " << result.y << std::endl;

//     return 0;
// }
