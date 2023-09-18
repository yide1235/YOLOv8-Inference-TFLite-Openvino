


#include <iostream>
// #include <vector>
// #include <fstream>
// #include "tensorflow/lite/interpreter.h"
// #include "tensorflow/lite/kernels/register.h"

// #include <tensorflow/lite/interpreter.h>
// #include <tensorflow/lite/model.h>

// #include <tensorflow/lite/kernels/register.h>
// #include <tensorflow/lite/optional_debug_tools.h>

// #include <tensorflow/lite/delegates/nnapi/nnapi_delegate.h>
// #include <tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h>
// #include <tensorflow/lite/tools/command_line_flags.h>
// #include <tensorflow/lite/tools/delegates/delegate_provider.h>
// #include <tensorflow/lite/tools/evaluation/utils.h>

// #include <tensorflow/lite/delegates/external/external_delegate.h>

// #include "tensorflow/lite/interpreter.h"
// #include <tensorflow/lite/kernels/register.h>
// #include <tensorflow/lite/model.h>
// #include <tensorflow/lite/optional_debug_tools.h>
// #include <tensorflow/lite/minimal_logging.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <cmath>


using namespace std;
using namespace cv;

//g++ -o yolov8_integer_tracking yolov8_integer_tracking.cpp `pkg-config --cflags --libs opencv4`
//./yolov8_integer_tracking

//simply copy objects here
std::vector<std::vector<float>> output_id(std::string img_path, cv::Mat results){

    cv::Mat img = cv::imread(img_path);
    if (img.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        // You should return an empty cv::Mat or handle errors differently.
        return cv::Mat();
    }

    //must assume detected image and results are not 0 in width and height
    std::vector<std::vector<float>> unique_ids;


    int len_results=results.rows;

    // cv::Mat unique_ids= cv::Mat::zeros(len_results, 27, CV_32F);
    //27 is ((10*(cls_id), b,g,r, confidence*100, x1/3,y1/3,x2/3,y2/3, b_detected/45, g_detected/45, r_detected/45))
    // 1,6,6,6,1,4,3,   

    for (int i =0; i< len_results; ++i){
        int cls_id=results.at<float>(i, 5);
        float confidence = results.at<float>(i, 4);

        vector <float> x;

        for (int j=0; j< 4; ++j){
            int each = static_cast<int>(std::round(results.at<float>(i, j)));
            x.push_back(each);
        }

        cv::Mat detected=img(cv::Rect(x[0], x[1], x[2] - x[0], x[3] - x[1]));
        // cv::namedWindow("Detected Region", cv::WINDOW_NORMAL);
        // cv::imshow("Detected Region", detected);

        // // Wait for a key press and then close the window
        // cv::waitKey(0);
        // cv::destroyAllWindows();

        //detected.cols is height .shape[1]
        //detected.rows is width  .shape[0]
        // std::cout << detected.cols << std::endl;
        // std::cout << detected.rows << std::endl;

        if (detected.rows == 0 || detected.cols == 0) {
            // Report an error or handle the case where either rows or cols is 0
            std::cerr << "Error: detected.rows or detected.cols is 0." << std::endl;
            // Optionally, you can exit the program or take appropriate action here.
        }

        
        int split=2;
        float block_width=detected.rows/split;
        float block_height=detected.cols/split;

        // Create a vector to store the blocks
        std::vector<cv::Mat> blocks;

        // Split the image into blocks
        for (int i = 0; i < split; ++i) {
            for (int j = 0; j < split; ++j) {
                int x1 = j * block_width;
                int y1 = i * block_height;
                int x2 = (j + 1) * block_width;
                int y2 = (i + 1) * block_height;

                // Ensure the coordinates are within bounds
                x2 = std::min(x2, detected.rows);
                y2 = std::min(y2, detected.cols);

                // Extract the block
                cv::Mat block = detected(cv::Rect(y1, x1, y2 - y1, x2 - x1));
                // cv::imshow("Detected Region", block);

                // // Wait for a key press and then close the window
                // cv::waitKey(0);
                // cv::destroyAllWindows();

                blocks.push_back(block);
            }
        }

        // Separate B, G, and R channels of each block


        std::vector<cv::Mat> b, g, r;
        for (const cv::Mat& block : blocks) {
            std::vector<cv::Mat> channels(3);
            cv::split(block, channels);
            
            b.push_back(channels[0]);
            g.push_back(channels[1]);
            r.push_back(channels[2]);
        }

        // Calculate variance and sort for each channel
        std::vector<int> b_var, g_var, r_var;

        for (size_t i = 0; i < b.size(); ++i) {
            
            for (size_t j = i + 1; j < b.size(); ++j) {

                cv::Scalar bi_mean, gi_mean, ri_mean;
                cv::Scalar bi_stddev, gi_stddev, ri_stddev;
                cv::Scalar bj_mean, gj_mean, rj_mean;
                cv::Scalar bj_stddev, gj_stddev, rj_stddev;


                cv::meanStdDev(b[i], bi_mean, bi_stddev);
                cv::meanStdDev(b[j], bj_mean, bj_stddev);
                cv::meanStdDev(g[i], gi_mean, gi_stddev);
                cv::meanStdDev(g[j], gj_mean, gj_stddev);
                cv::meanStdDev(r[i], ri_mean, ri_stddev);
                cv::meanStdDev(r[j], rj_mean, rj_stddev);


                double bi_var = bi_stddev[0] * bi_stddev[0];
                double bj_var = bj_stddev[0] * bj_stddev[0];
                double b_var_result=bi_var*bj_var;

                double gi_var = gi_stddev[0] * gi_stddev[0];
                double gj_var = gj_stddev[0] * gj_stddev[0];
                double g_var_result=gi_var*gj_var;

                double ri_var = ri_stddev[0] * ri_stddev[0];
                double rj_var = rj_stddev[0] * rj_stddev[0];
                double r_var_result=ri_var*rj_var;


                b_var.push_back(static_cast<int>(b_var_result / (detected.rows * detected.cols)));
                g_var.push_back(static_cast<int>(g_var_result / (detected.rows * detected.cols)));
                r_var.push_back(static_cast<int>(r_var_result / (detected.rows * detected.cols)));


            }
        }

        // Sort the variance vectors
        std::sort(b_var.begin(), b_var.end());
        std::sort(g_var.begin(), g_var.end());
        std::sort(r_var.begin(), r_var.end());


        // // Printing b_var
        // std::cout << "b_var: ";
        // for (int i = 0; i < b_var.size(); ++i) {
        //     std::cout << b_var[i] << " ";
        // }
        // std::cout << std::endl;

        // // Printing g_var
        // std::cout << "g_var: ";
        // for (int i = 0; i < g_var.size(); ++i) {
        //     std::cout << g_var[i] << " ";
        // }
        // std::cout << std::endl;

        // // Printing r_var
        // std::cout << "r_var: ";
        // for (int i = 0; i < r_var.size(); ++i) {
        //     std::cout << r_var[i] << " ";
        // }
        // std::cout << std::endl;


        //so b_var, g_var, r_var is std::vector<int> b_var, g_var, r_var;

        // Calculate the maximum and minimum values for b, g, and r
        double b_max = *std::max_element(b_var.begin(), b_var.end());
        double b_min = *std::min_element(b_var.begin(), b_var.end());

        double g_max = *std::max_element(g_var.begin(), g_var.end());
        double g_min = *std::min_element(g_var.begin(), g_var.end());

        double r_max = *std::max_element(r_var.begin(), r_var.end());
        double r_min = *std::min_element(r_var.begin(), r_var.end());

        // Calculate intervals
        double b_interval = (b_max - b_min) / (b_var.size() - 1);
        double g_interval = (g_max - g_min) / (g_var.size() - 1);
        double r_interval = (r_max - r_min) / (r_var.size() - 1);

        // Normalize the values in b_var, g_var, and r_var
        for (size_t i = 0; i < b_var.size(); ++i) {
            if (b_interval != 0) {
                b_var[i] = (b_var[i] - b_min) / b_interval + 1;
            } else {
                // Handle the case where b_interval is zero (or any other invalid value)
                // You can assign a default value or raise an exception depending on your logic.
                b_var[i] = 0;
            }
        }

        for (size_t i = 0; i < g_var.size(); ++i) {
            if (g_interval != 0) {
                g_var[i] = (g_var[i] - g_min) / g_interval + 1;
            } else {
                g_var[i] = 0;
            }
        }

        for (size_t i = 0; i < r_var.size(); ++i) {
            if (r_interval != 0) {
                r_var[i] = (r_var[i] - r_min) / r_interval + 1;
            } else {
                r_var[i] = 0;
            }
        }


        cv::Mat b_detected;
        cv::Mat g_detected;
        cv::Mat r_detected;    

        std::vector<cv::Mat> channels_detected(3);
        cv::split(detected, channels_detected);
        
        b_detected=channels_detected[0];
        g_detected=channels_detected[1];
        r_detected=channels_detected[2];


        cv::Scalar b_mean, g_mean, r_mean;
        cv::Scalar b_stddev, g_stddev, r_stddev;

        cv::meanStdDev(b_detected, b_mean, b_stddev);
        cv::meanStdDev(g_detected, g_mean, g_stddev);
        cv::meanStdDev(r_detected, r_mean, r_stddev);
        //just use b_mean/45, g_mean/45, r_mean/45

        std::vector<float> combinedData;
        combinedData.push_back(static_cast<float>(cls_id*10));

        for (const int& value : b_var) {
            combinedData.push_back(static_cast<float>(value));
        }
        for (const int& value : g_var) {
            combinedData.push_back(static_cast<float>(value));
        }
        for (const int& value : r_var) {
            combinedData.push_back(static_cast<float>(value));
        }

        combinedData.push_back(confidence*100);

        combinedData.push_back(x[0]/3);
        combinedData.push_back(x[1]/3);
        combinedData.push_back(x[2]/3);
        combinedData.push_back(x[3]/3);

        // std::cout << b_detected.rows << std::endl;
        // std::cout << b_detected.cols << std::endl;

        combinedData.push_back(static_cast<float>(b_stddev[0]*b_stddev[0]/45));
        combinedData.push_back(static_cast<float>(g_stddev[0]*g_stddev[0]/45));
        combinedData.push_back(static_cast<float>(r_stddev[0]*r_stddev[0]/45));

        // std::cout << b_mean[0] << std::endl;
        // std::cout << g_mean[0] << std::endl;
        // std::cout << r_mean[0] << std::endl;


        // std::cout << "combinedData: ";
        // for (size_t i = 0; i < combinedData.size(); ++i) {
        //     std::cout << combinedData[i] << " ";
        // }
        // std::cout << std::endl;
        unique_ids.push_back(combinedData);

    }//end of for loop



    // std::cout << "unique_id: ";
    // for (size_t i = 0; i < unique_ids.size(); ++i) {
    //     std::cout << unique_ids[i] << " ";
    // }
    // std::cout << std::endl;

    return unique_ids;

}





int main(){

    // Define the rows and columns of your matrices
    int rows = 14;
    int cols = 6;

    // Create two empty matrices of the desired size and type (float)
    cv::Mat matrix1(rows, cols, CV_32F);
    cv::Mat matrix2(rows, cols, CV_32F);

    // Assign the values to the first matrix
    float values1[14][6] = {
        {8.89233948e+02, 1.40315643e+02, 9.93849792e+02, 4.13926117e+02, 8.08927953e-01, 0.00000000e+00},
        {0.00000000e+00, 2.77120880e+02, 6.43789291e+01, 4.54162903e+02, 6.70613885e-01, 0.00000000e+00},
        {1.93136780e+02, 2.57002441e+02, 3.05799896e+02, 5.22565613e+02, 8.08927953e-01, 0.00000000e+00},
        {4.02368546e+00, 5.62802368e+02, 7.64499741e+01, 7.15702332e+02, 2.76628226e-01, 0.00000000e+00},
        {1.19905774e+03, 2.20789307e+02, 1.39219434e+03, 5.26589172e+02, 8.71798038e-01, 0.00000000e+00},
        {3.37989380e+02, 4.01855072e+02, 5.95505127e+02, 8.12270691e+02, 8.38267326e-01, 0.00000000e+00},
        {7.36333984e+02, 6.23157593e+02, 1.09846545e+03, 8.80673279e+02, 8.71798038e-01, 0.00000000e+00},
        {7.72547119e+02, 2.16765656e+02, 1.06225232e+03, 3.85760284e+02, 4.40090358e-01, 1.30000000e+01},
        {1.46462073e+03, 4.30020813e+02, 1.73823096e+03, 6.87536499e+02, 8.08927953e-01, 1.30000000e+01},
        {6.19647217e+02, 7.23749695e+02, 1.34391003e+03, 1.05369177e+03, 8.08927953e-01, 1.30000000e+01},
        {1.40828903e+02, 4.09902466e+02, 2.13255203e+02, 4.66234039e+02, 4.40090358e-01, 2.60000000e+01},
        {1.11456018e+03, 3.17357697e+02, 1.29964966e+03, 5.10494476e+02, 5.57447791e-01, 5.60000000e+01},
        {2.57515747e+02, 5.02447174e+02, 4.98936707e+02, 7.92152344e+02, 3.26924264e-01, 5.60000000e+01},
        {9.09352356e+02, 7.19725952e+02, 9.73731262e+02, 7.84104919e+02, 2.76628226e-01, 6.70000000e+01}
    };

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix1.at<float>(i, j) = values1[i][j];
        }
    }

    // Assign the values to the second matrix
    float values2[14][6] = {
        {8.97281311e+02, 1.40315643e+02, 9.85802368e+02, 4.30020813e+02, 7.67014623e-01, 0.00000000e+00},
        {4.02368355e+00, 2.77120880e+02, 6.03552475e+01, 4.54162903e+02, 7.67014623e-01, 0.00000000e+00},
        {1.93136780e+02, 2.61026123e+02, 3.05799896e+02, 5.34636658e+02, 8.08927953e-01, 0.00000000e+00},
        {8.04736710e+00, 5.66826050e+02, 1.04615761e+02, 6.79489197e+02, 4.40090358e-01, 0.00000000e+00},
        {1.19905774e+03, 2.44931442e+02, 1.39219434e+03, 5.34636536e+02, 8.71798038e-01, 0.00000000e+00},
        {5.11007751e+02, 3.69665588e+02, 6.96097168e+02, 7.47891785e+02, 8.08927953e-01, 0.00000000e+00},
        {7.36333984e+02, 6.23157593e+02, 1.09846545e+03, 8.80673279e+02, 8.71798038e-01, 0.00000000e+00},
        {7.80594543e+02, 2.16765656e+02, 1.08639441e+03, 3.85760284e+02, 2.76628226e-01, 1.30000000e+01},
        {1.44852588e+03, 4.30020813e+02, 1.72213635e+03, 6.87536499e+02, 7.67014623e-01, 1.30000000e+01},
        {6.35741943e+02, 7.47891785e+02, 1.36000488e+03, 1.07200000e+03, 8.08927953e-01, 1.30000000e+01},
        {1.40828903e+02, 4.09902466e+02, 2.13255203e+02, 4.66234039e+02, 3.26924264e-01, 2.60000000e+01},
        {1.11456018e+03, 3.09310364e+02, 1.29964966e+03, 5.02447174e+02, 4.98769075e-01, 5.60000000e+01},
        {2.57515747e+02, 5.02447174e+02, 4.98936707e+02, 7.92152344e+02, 3.26924264e-01, 5.60000000e+01},
        {9.09352356e+02, 7.19725952e+02, 9.73731262e+02, 7.84104919e+02, 4.40090358e-01, 6.70000000e+01}
    };


    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix2.at<float>(i, j) = values2[i][j];
        }
    }

    std::cout << "Matrix1:" << std::endl;
    std::cout << matrix1 << std::endl;

    std::cout << "Matrix2:" << std::endl;
    std::cout << matrix2 << std::endl;

    std::string img1="./test/test_case1/0300.jpg";
    std::string img2="./test/test_case1/0330.jpg";

    std::vector<std::vector<float>> output_id1=output_id(img1, matrix1);
    std::vector<std::vector<float>> output_id2=output_id(img2, matrix2);        


    // //Iterate through the outer vector
    // for (const std::vector<float>& innerVector : output_id1) {
    //     // Iterate through the inner vector
    //     for (float value : innerVector) {
    //         std::cout << value << " ";
    //     }
    //     // Print a newline to separate rows
    //     std::cout << std::endl;
    // }




    return 0;
}


// g++ -I../tensorflow -ltensorflow_cc -c simple.cpp `pkg-config --cflags --libs opencv4`


//output data is (1,84,8400)


// results is 
// [[8.89233948e+02 1.40315643e+02 9.93849792e+02 4.13926117e+02
//   8.08927953e-01 0.00000000e+00]
//  [0.00000000e+00 2.77120880e+02 6.43789291e+01 4.54162903e+02
//   6.70613885e-01 0.00000000e+00]
//  [1.93136780e+02 2.57002441e+02 3.05799896e+02 5.22565613e+02
//   8.08927953e-01 0.00000000e+00]
//  [4.02368546e+00 5.62802368e+02 7.64499741e+01 7.15702332e+02
//   2.76628226e-01 0.00000000e+00]
//  [1.19905774e+03 2.20789307e+02 1.39219434e+03 5.26589172e+02
//   8.71798038e-01 0.00000000e+00]
//  [3.37989380e+02 4.01855072e+02 5.95505127e+02 8.12270691e+02
//   8.38267326e-01 0.00000000e+00]
//  [7.36333984e+02 6.23157593e+02 1.09846545e+03 8.80673279e+02
//   8.71798038e-01 0.00000000e+00]
//  [7.72547119e+02 2.16765656e+02 1.06225232e+03 3.85760284e+02
//   4.40090358e-01 1.30000000e+01]
//  [1.46462073e+03 4.30020813e+02 1.73823096e+03 6.87536499e+02
//   8.08927953e-01 1.30000000e+01]
//  [6.19647217e+02 7.23749695e+02 1.34391003e+03 1.05369177e+03
//   8.08927953e-01 1.30000000e+01]
//  [1.40828903e+02 4.09902466e+02 2.13255203e+02 4.66234039e+02
//   4.40090358e-01 2.60000000e+01]
//  [1.11456018e+03 3.17357697e+02 1.29964966e+03 5.10494476e+02
//   5.57447791e-01 5.60000000e+01]
//  [2.57515747e+02 5.02447174e+02 4.98936707e+02 7.92152344e+02
//   3.26924264e-01 5.60000000e+01]
//  [9.09352356e+02 7.19725952e+02 9.73731262e+02 7.84104919e+02
//   2.76628226e-01 6.70000000e+01]]
//--------------------------------------------------------------
// [[8.97281311e+02 1.40315643e+02 9.85802368e+02 4.30020813e+02
//   7.67014623e-01 0.00000000e+00]
//  [4.02368355e+00 2.77120880e+02 6.03552475e+01 4.54162903e+02
//   7.67014623e-01 0.00000000e+00]
//  [1.93136780e+02 2.61026123e+02 3.05799896e+02 5.34636658e+02
//   8.08927953e-01 0.00000000e+00]
//  [8.04736710e+00 5.66826050e+02 1.04615761e+02 6.79489197e+02
//   4.40090358e-01 0.00000000e+00]
//  [1.19905774e+03 2.44931442e+02 1.39219434e+03 5.34636536e+02
//   8.71798038e-01 0.00000000e+00]
//  [5.11007751e+02 3.69665588e+02 6.96097168e+02 7.47891785e+02
//   8.08927953e-01 0.00000000e+00]
//  [7.36333984e+02 6.23157593e+02 1.09846545e+03 8.80673279e+02
//   8.71798038e-01 0.00000000e+00]
//  [7.80594543e+02 2.16765656e+02 1.08639441e+03 3.85760284e+02
//   2.76628226e-01 1.30000000e+01]
//  [1.44852588e+03 4.30020813e+02 1.72213635e+03 6.87536499e+02
//   7.67014623e-01 1.30000000e+01]
//  [6.35741943e+02 7.47891785e+02 1.36000488e+03 1.07200000e+03
//   8.08927953e-01 1.30000000e+01]
//  [1.40828903e+02 4.09902466e+02 2.13255203e+02 4.66234039e+02
//   3.26924264e-01 2.60000000e+01]
//  [1.11456018e+03 3.09310364e+02 1.29964966e+03 5.02447174e+02
//   4.98769075e-01 5.60000000e+01]
//  [2.57515747e+02 5.02447174e+02 4.98936707e+02 7.92152344e+02
//   3.26924264e-01 5.60000000e+01]
//  [9.09352356e+02 7.19725952e+02 9.73731262e+02 7.84104919e+02
//   4.40090358e-01 6.70000000e+01]]
