#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

cv::Mat triangulatePoint(const cv::Mat& M_r, const cv::Mat& P_l, const cv::Point2f& point_l, const cv::Point2f& point_r) {
    // used in below function, should never need to be called individually, see below for inputs
    cv::Mat A_b(4, 3, CV_64F);
    A_b.at<double>(0, 0) = point_r.x * M_r.at<double>(2, 0) - M_r.at<double>(0, 0);
    A_b.at<double>(0, 1) = point_r.x * M_r.at<double>(2, 1) - M_r.at<double>(0, 1);
    A_b.at<double>(0, 2) = point_r.x * M_r.at<double>(2, 2) - M_r.at<double>(0, 2);
    A_b.at<double>(1, 0) = point_r.y * M_r.at<double>(2, 0) - M_r.at<double>(1, 0);
    A_b.at<double>(1, 1) = point_r.y * M_r.at<double>(2, 1) - M_r.at<double>(1, 1);
    A_b.at<double>(1, 2) = point_r.y * M_r.at<double>(2, 2) - M_r.at<double>(1, 2);
    A_b.at<double>(2, 0) = point_l.x * P_l.at<double>(2, 0) - P_l.at<double>(0, 0);
    A_b.at<double>(2, 1) = point_l.x * P_l.at<double>(2, 1) - P_l.at<double>(0, 1);
    A_b.at<double>(2, 2) = point_l.x * P_l.at<double>(2, 2) - P_l.at<double>(0, 2);
    A_b.at<double>(3, 0) = point_l.y * P_l.at<double>(2, 0) - P_l.at<double>(1, 0);
    A_b.at<double>(3, 1) = point_l.y * P_l.at<double>(2, 1) - P_l.at<double>(1, 1);
    A_b.at<double>(3, 2) = point_l.y * P_l.at<double>(2, 2) - P_l.at<double>(1, 2);
    
    cv::Mat b(4, 1, CV_64F);
    b.at<double>(0, 0) = M_r.at<double>(0, 3) - M_r.at<double>(2, 3);
    b.at<double>(1, 0) = M_r.at<double>(1, 3) - M_r.at<double>(2, 3);
    b.at<double>(2, 0) = P_l.at<double>(0, 3) - P_l.at<double>(2, 3);
    b.at<double>(3, 0) = P_l.at<double>(1, 3) - P_l.at<double>(2, 3);
    
    cv::Mat X;
    try {
        cv::Mat A_b_transpose = A_b.t();
        cv::Mat A_b_product = A_b_transpose * A_b;
        cv::Mat A_b_inv = A_b_product.inv();
        X = A_b_inv * A_b_transpose * b;
    } catch (const cv::Exception& e) {
        cv::Mat A_b_transpose = A_b.t();
        cv::Mat A_b_product = A_b_transpose * A_b;
        cv::Mat A_b_pseudoinv = A_b_product.inv(cv::DECOMP_SVD);
        X = A_b_pseudoinv * A_b_transpose * b;
    }
    
    return X;
}

std::vector<cv::Point3f> triangulatePoints(const cv::Mat& K_l, const cv::Mat& K_r, const cv::Mat& R, const cv::Mat& T,
                                           const std::vector<std::vector<cv::Point2f>>& uvs_l, const std::vector<std::vector<cv::Point2f>>& uvs_r) {
    // takes in: intrinsic matrix for left camera (K_l), right camera intrinsic matrix (K_r), extrinsic rotation matrix (R), extrinsic translation matrix (T)
    // uvs_l list of list of 2d points for left camera, uvs_r is the list of list of 2d points in right camera
    // returns list of list of 3d points
    // Create matrices for M_l and M_r
    cv::Mat M_l = cv::Mat::zeros(3, 4, CV_64F);
    cv::Mat M_r = cv::Mat::zeros(3, 4, CV_64F);

    // Set values for M_l and M_r
    K_l.copyTo(M_l(cv::Rect(0, 0, 3, 3)));
    K_r.copyTo(M_r(cv::Rect(0, 0, 3, 3)));

    cv::Mat RT = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat R_with_T = cv::Mat::eye(4, 4, CV_64F);
    R.copyTo(R_with_T(cv::Rect(0, 0, 3, 3)));
    T.copyTo(R_with_T(cv::Rect(3, 0, 1, 3)));
    RT = R_with_T;
    cv::Mat P_l = M_l * RT;

    std::vector<cv::Point3f> p3d_cam;
    for (size_t i = 0; i < uvs_l.size(); ++i) {
        for (size_t j = 0; j < uvs_l[i].size(); ++j) {
            cv::Mat point = triangulatePoint(M_r, P_l, uvs_l[i][j], uvs_r[i][j]);
            p3d_cam.emplace_back(point.at<double>(0, 0), point.at<double>(1, 0), point.at<double>(2, 0));
        }
    }

    return p3d_cam;
}

// int main() {
//     // Define your matrices
//     cv::Mat K_l = (cv::Mat_<double>(3, 3) << 1381.35935,0,941.40662,
//                                              0,1381.35935*1.01825,470.19631,
//                                              0, 0, 1);

//     cv::Mat K_r = (cv::Mat_<double>(3, 3) << 1377.64593,0,889.06106,
//                                              0,1377.64593*1.01796,499.96914,
//                                              0, 0, 1);

//     cv::Mat R = (cv::Mat_<double>(3, 3) << 1.0, 0.0, 0.0,
//                                            0.0, 1.0, 0.0,
//                                            0.0, 0.0, 1.0);
//     std::cout<<R<<std::endl;

//     cv::Mat T = (cv::Mat_<double>(3, 1) << 65.0, 0.0, 0.0);

//     // Generate random uvs_l and uvs_r points for testing
//     std::vector<std::vector<cv::Point2f>> uvs_l, uvs_r;
//     int num_rows = 1;
//     int num_cols = 1;
//     for (int i = 0; i < num_rows; ++i) {
//         std::vector<cv::Point2f> row_l, row_r;
//         for (int j = 0; j < num_cols; ++j) {
//             row_l.emplace_back(100.0, 200.0); // Random points for uvs_l
//             row_r.emplace_back(60.0, 220.0); // Random points for uvs_r
//         }
//         uvs_l.push_back(row_l);
//         uvs_r.push_back(row_r);
//     }

//         std::vector<cv::Point3f> p3d_cam = triangulatePoints(K_l, K_r, R, T, uvs_l, uvs_r);

//     // Print or use the triangulated 3D points
//     for (const cv::Point3f& point : p3d_cam) {
//         std::cout << "Triangulated Point: (" << point.x << ", " << point.y << ", " << point.z << ")\n";
//     }

//     return 0;
// }







