#include <iostream>
#include <vector>
#include <opencv4/opencv2/opencv.hpp>
#include "../../include/point_2d_to_3d/line_matching_4.h"
#include "../../include/point_2d_to_3d/triangulate_point.h"

std::vector<cv::Point3f> _2d_point_to_3d_point(const cv::Mat& K_l, const cv::Mat& K_r, const cv::Mat& R, const cv::Mat& T, const cv::Mat& H1, const cv::Mat& H2, const cv::Point2f& p1, const cv::Mat& img1, const cv::Mat& img2) {
    // Takes in left camera intrinsic matrix (K_l), right camera intrinsic matrix (K_r), extrinsic rotation matrix (R), extrinsic translation matrix (T),
    //     left camera rectification homography (H1), right camera rectification homography (H2), point in left image (p1), left image (img1), right image (img2)
    //  intrinsics, extrinsics, and homographies can be found in settings.xml file
    // images are to be rectified (can use cv::warpperspective function) in cv::Mat format
    // point is to be 2d float point
    
    int width_rad = 10;
    int height_rad = 10;
    cv::Point2i p2 = sweep_line_block(img1, img2, p1, width_rad, height_rad);

    cv::Mat H1_inverse;
    cv::invert(H1, H1_inverse, cv::DECOMP_LU);  
    cv::Mat H2_inverse;
    cv::invert(H2, H2_inverse, cv::DECOMP_LU);
    
    cv::Mat p1_mat = (cv::Mat_<double>(3, 1) << p1.x, p1.y, 1.0f);
    cv::Mat pp1_mat = H1_inverse * p1_mat;
    cv::Point2f pp1(pp1_mat.at<double>(0) / pp1_mat.at<double>(2), pp1_mat.at<double>(1) / pp1_mat.at<double>(2));

    cv::Mat p2_mat = (cv::Mat_<double>(3, 1) << p2.x, p2.y, 1.0f);
    cv::Mat pp2_mat = H2_inverse * p2_mat;
    cv::Point2f pp2(pp2_mat.at<double>(0) / pp2_mat.at<double>(2), pp2_mat.at<double>(1) / pp2_mat.at<double>(2));


    std::vector<std::vector<cv::Point2f>> tp2(1, std::vector<cv::Point2f>(1, pp2));
    std::vector<std::vector<cv::Point2f>> tp1(1, std::vector<cv::Point2f>(1, pp1));

    std::cout<<"tps:"<<std::endl<<pp1<<pp2<<std::endl<<std::endl;

    std::vector<cv::Point3f> finalResult = triangulatePoints(K_l, K_r, R, T, tp1, tp2);

    return finalResult;
}

int main() {

    cv::Mat H1, H2, K_l, K_r, R, T;
    cv::Point2f p1;
    cv::Mat img1, img2;

    K_l = (cv::Mat_<double>(3, 3) << 1381.35935, 0, 941.40662,
                                    0, 1381.35935 * 1.01825, 470.19631,
                                    0, 0, 1);

    K_r = (cv::Mat_<double>(3, 3) << 1377.64593, 0, 889.06106,
                                    0, 1377.64593 * 1.01796, 499.96914,
                                    0, 0, 1);

    R = (cv::Mat_<double>(3, 3) << 1.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0,
                                 0.0, 0.0, 1.0);

    T = (cv::Mat_<double>(3, 1) << 65.0, 0.0, 0.0);

    H1 = (cv::Mat_<double>(3, 3) << 4.22772245e-02, -1.30516921e-02, -8.08973235e+00,
                                  1.94229434e-02,  2.14953874e-02, -1.53770719e+01,
                                  1.83075763e-05, -8.81994917e-06, 1.42816432e-02);

    H2 = (cv::Mat_<double>(3, 3) << 1.59147194e+00, -5.66628608e-01, -2.61833611e+02,
                                  7.00703672e-01, 8.12012879e-01, -5.71162480e+02,
                                  6.76459965e-04, -2.40847205e-04, 4.80655924e-01);

    p1 = cv::Point2f(1000, 1000);

    img1 = cv::imread("../../Dropbox/REALTIME7/tmpl.jpg");
    img2 = cv::imread("../../Dropbox/REALTIME7/tmpr.jpg");

    std::vector<cv::Point3f> result = _2d_point_to_3d_point(K_l, K_r, R, T, H1, H2, p1, img1, img2);
    
    for (const auto& point : result) {
        std::cout << point << std::endl;
    }

    return 0;
}
