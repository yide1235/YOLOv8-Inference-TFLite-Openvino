


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
#include <map>
#include <string>

using namespace std;
using namespace cv;

//g++ -o yolov8_integer_tracking yolov8_integer_tracking.cpp `pkg-config --cflags --libs opencv4`
//./yolov8_integer_tracking

//simply copy objects here
std::vector<std::vector<float>> output_id(std::string& img_path, cv::Mat& results){

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








//start the second function:

//not correct valid
cv::Mat normalize(const cv::Mat& image) {
    cv::Mat im(image.rows, image.cols, CV_32FC3);

    for (int x = 0; x < image.rows; ++x) {
        for (int y = 0; y < image.cols; ++y) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(x, y);
            float div = std::max(std::max(pixel[0], pixel[1]), pixel[2]);

            if (div == 0) {
                div = 1;
            }

            cv::Vec3f normalized_pixel;
            normalized_pixel[0] = static_cast<float>(pixel[0]) / div;
            normalized_pixel[1] = static_cast<float>(pixel[1]) / div;
            normalized_pixel[2] = static_cast<float>(pixel[2]) / div;

            im.at<cv::Vec3f>(x, y) = normalized_pixel;
        }
    }

    return im;
}

// Define a function to calculate SVD (you should implement this function)
std::vector<cv::Mat> calculateSVD(const cv::Mat& detected) {
    cv::Mat l1, l2, l3;
    cv::Mat l4;

    cv::Mat image1 = normalize(detected);
    int height = image1.rows;
    int width = image1.cols;

    cv::Mat rc = image1.clone();
    cv::Mat gc = image1.clone();
    cv::Mat bc = image1.clone();

    // Extract color channels
    cv::Mat channels[3];
    cv::split(image1, channels);
    rc = channels[0];
    gc = channels[1];
    bc = channels[2];

    // Perform SVD on each channel
    cv::SVD svd_rc(rc, cv::SVD::FULL_UV);
    cv::SVD svd_gc(gc, cv::SVD::FULL_UV);
    cv::SVD svd_bc(bc, cv::SVD::FULL_UV);

    // Get the first 5 singular values for each channel
    cv::Mat s_rc, s_gc, s_bc;


    if (svd_rc.w.rows >= 5) {
        s_rc = svd_rc.w.rowRange(0, 5);
        l1=s_rc;
    } else {
        // Handle the case when the matrix doesn't have enough rows
        // You may want to add error handling or return empty matrices here
    }

    if (svd_gc.w.rows >= 5) {
        s_gc = svd_gc.w.rowRange(0, 5);
        l2=s_gc;
    } else {
        // Handle the case when the matrix doesn't have enough rows
        // You may want to add error handling or return empty matrices here
    }

    if (svd_bc.w.rows >= 5) {
        s_bc = svd_bc.w.rowRange(0, 5);
        l3=s_bc;
    } else {
        // Handle the case when the matrix doesn't have enough rows
        // You may want to add error handling or return empty matrices here
    }

    // Store image dimensions in l4
    l4= cv::Mat(1, 2, CV_32S);
    l4.at<int>(0, 0) = height;
    l4.at<int>(0, 1) = width;

    std::vector<cv::Mat> result;

    result.push_back(l1);
    result.push_back(l2);
    result.push_back(l3);
    result.push_back(l4);



    return result;

}


double get_score(
    const cv::Mat& l1, const cv::Mat& l2, const cv::Mat& l3, const cv::Mat& l4,
    const cv::Mat& c1, const cv::Mat& c2, const cv::Mat& c3, const cv::Mat& c4) {
    
    float sum = 0;
    float mag2 = 0;
    // int mag2_int=0;

    // std::cout << l1.rows << std::endl;
    // std::cout << l1.cols << std::endl;

    for (int x = 0; x < l1.rows; ++x) {
        // std::cout << l1.at<float>(x, 0) << std::endl;
        sum += pow(pow(l1.at<float>(x, 0) - c1.at<float>(x, 0), 2) +
                    pow(l2.at<float>(x,0) - c2.at<float>(x,0), 2) +
                    pow(l3.at<float>(x,0) - c3.at<float>(x,0), 2),0.5);
        // std::cout << sum << std::endl;
    }

    // std::cout << l4.rows << std::endl;
    // std::cout << l4.cols << std::endl;
    // std::cout << l4 << std::endl;
    // std::cout << l4.at<int>(0, 0) << std::endl;
    // std::cout << l4.at<int>(0, 1) << std::endl;
    for (int x = 0; x < l4.rows; ++x) {
        // std::cout << static_cast<float>(l4.at<int>(0, x)) << std::endl;

        mag2 += (pow(static_cast<float>(l4.at<int>(0, x)) - static_cast<float>(c4.at<int>(0, x)), 2));
        // mag2_int+= pow(l4.at<int>(0, x) - c4.at<int>(0, x), 2);
    }


    std::cout << mag2<< std::endl;

    mag2 = mag2 / pow((static_cast<float>(l4.at<int>(0, 0))  * static_cast<float>(l4.at<int>(0, 1)) )
     + (static_cast<float>(c4.at<int>(0, 0)) * static_cast<float>(c4.at<int>(0, 1))), 0.5);
    std::cout << mag2 << std::endl;
    float mag1 = pow(sum, 0.5);
    mag2 = pow(mag2, 0.5);

    // std::cout << mag1 << std::endl;
    // std::cout << mag2 << std::endl;
    return mag1 + mag2;
    // return 1;
}



std::map<int, std::vector<float>> generateIds(const std::vector<std::vector<float>>& unique_ids) {
    std::map<int, std::vector<float>> ids;
    for (int i = 0; i < unique_ids.size(); ++i) {
        ids[i] = {static_cast<float>(i), -1.0f, static_cast<float>(unique_ids[i][0] / 10.0)};
    }
    return ids;
}

std::vector<std::map<int, std::vector<float>>> compare(
    std::string& img_path1, cv::Mat& results1,
    const std::vector<std::vector<float>>& unique_ids1,
    std::string& img_path2, cv::Mat& results2,
    const std::vector<std::vector<float>>& unique_ids2) 
{

    std::vector<std::map<int, std::vector<float>>> result;

    cv::Mat image1 = cv::imread(img_path1);
    if (image1.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        // You should return an empty cv::Mat or handle errors differently.
        return std::vector<std::map<int, std::vector<float>>>();
    }

    cv::Mat image2 = cv::imread(img_path2);
    if (image2.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        // You should return an empty cv::Mat or handle errors differently.
        return std::vector<std::map<int, std::vector<float>>>();
    }

    int svd_threshold = 20;
    int cut_threshold = 100;

    std::map<int, std::vector<float>> ids1;
    std::map<int, std::vector<float>> ids2;
    
    if (unique_ids1.size() > unique_ids2.size()) {

    }
    else {
        ids2 = generateIds(unique_ids2);

        int addition = 1;

        // // Iterate through the map and print the key-value pairs
        // for (const auto& entry : ids2) {
        //     int key = entry.first;
        //     const std::vector<float>& values = entry.second;

        //     std::cout << "Key: " << key << ", Values: ";
            
        //     for (float value : values) {
        //         std::cout << value << " ";
        //     }

        //     std::cout << std::endl;
        // }        

        
        // Iterate through the vectors in unique_ids1
        for (size_t i = 0; i < unique_ids1.size(); ++i) {
            float min_norm1 = std::numeric_limits<float>::infinity();
            int matching_id = -1;

            // Compare with vectors in unique_ids2
            for (size_t j = 0; j < unique_ids2.size(); ++j) {
                if (unique_ids1[i][0] == unique_ids2[j][0]) {
                    float norm = 0.0;
                    for (size_t k = 1; k < unique_ids1[i].size(); ++k) {
                        norm += std::pow(unique_ids2[j][k] - unique_ids1[i][k], 2);
                    }
                    norm = std::sqrt(norm);

                    if (norm < min_norm1) {
                        min_norm1 = norm;
                        matching_id = static_cast<int>(j);
                    }
                }
            }

            if (cut_threshold > min_norm1) {
                ids1[i] = {matching_id, min_norm1, unique_ids1[i][0] / 10};
                ids2[matching_id][1] = 1.0;
            } else {
                ids1[i] = {-1, -1, unique_ids1[i][0] / 10};
            }
        }


        // ids1[0][0]=3;



        // // Iterate through the map and print the key-value pairs
        // for (const auto& entry : ids1) {
        //     int key = entry.first;
        //     const std::vector<float>& values = entry.second;

        //     std::cout << "Key: " << key << ", Values: ";
            
        //     for (float value : values) {
        //         std::cout << value << " ";
        //     }

        //     std::cout << std::endl;
        // }    

        // // Iterate through the map and print the key-value pairs
        // for (const auto& entry : ids2) {
        //     int key = entry.first;
        //     const std::vector<float>& values = entry.second;

        //     std::cout << "Key: " << key << ", Values: ";
            
        //     for (float value : values) {
        //         std::cout << value << " ";
        //     }

        //     std::cout << std::endl;
        // }    

        
        


        for (auto it1 = ids1.begin(); it1 != ids1.end(); ++it1) {
            for (auto it2 = ids1.begin(); it2 != ids1.end(); ++it2) {
                int key1 = it1->first;
                int key2 = it2->first;
                std::vector<float>& value1 = it1->second;
                std::vector<float>& value2 = it2->second;

                if (key1 != key2 && value1[0] == value2[0]) {
                    if (value1[2] == value2[2]) {
                        if (value1[1] != -1 && value2[1] != -1) {
                            if (value1[1] > value2[1]) {
                                value2[0] = ids1.size() + addition;
                                addition++;
                            } else {
                                value1[0] = ids1.size() + addition;
                                addition++;
                            }
                        }
                    } else {
                        if (value1[1] != -1 && value2[1] != -1) {
                            value1[0] = ids1.size() + addition + 1;
                            addition++;
                        }
                    }
                }
            }
        }

        // std::cout << "111111111111111111111" << std::endl;


        // // Iterate through the map and print the key-value pairs
        // for (const auto& entry : ids1) {
        //     int key = entry.first;
        //     const std::vector<float>& values = entry.second;

        //     std::cout << "Key: " << key << ", Values: ";
            
        //     for (float value : values) {
        //         std::cout << value << " ";
        //     }

        //     std::cout << std::endl;
        // }    

        // // Iterate through the map and print the key-value pairs
        // for (const auto& entry : ids2) {
        //     int key = entry.first;
        //     const std::vector<float>& values = entry.second;

        //     std::cout << "Key: " << key << ", Values: ";
            
        //     for (float value : values) {
        //         std::cout << value << " ";
        //     }

        //     std::cout << std::endl;
        // }    

        for (size_t i = 0; i < ids1.size(); ++i) {
            if (ids1[i][0] == -1) {

                vector <float> x;

                for (int g=0; g< 4; ++g){
                    int each = static_cast<int>(std::round(results1.at<float>(i, g)));
                    x.push_back(each);
                }

                cv::Mat detected1 = image1(cv::Rect(x[0], x[1], x[2] - x[0], x[3] - x[1]));

                float class_id1 = results1.at<float>(i, 5);

                int index = -1;
                float min_score = 1000000.0;

                for (size_t j = 0; j < ids2.size(); ++j) {
                    if (ids2[j][1] == -1) {
                        // std::cout << "------------" << std::endl;
                        vector <float> x2;
                        for (int k=0; k< 4; ++k){
                            int each2 = static_cast<int>(std::round(results2.at<float>(j, k)));
                            x2.push_back(each2);
                        }

                        cv::Mat detected2 = image2(cv::Rect(x2[0], x2[1], x2[2] - x2[0], x2[3] - x2[1]));

                        float class_id2 = results2.at<float>(j, 5);

                        std::vector<cv::Mat> l1l4 = calculateSVD(detected1);

                    //    for (size_t i = 0; i < l1l4.size(); ++i) {
                    //         std::cout << "Matrix " << i << ":" << std::endl;
                    //         std::cout << l1l4[i] << std::endl;
                    //     }

                        // std::cout << "------------" << std::endl;
                        

                        if (class_id1 == class_id2){
                            std::vector<cv::Mat> c1c4 = calculateSVD(detected2);

                                //it is correct now for both l1l4 and c1c4

                        //     for (size_t i = 0; i < c1c4.size(); ++i) {
                        //             std::cout << "Matrix " << i << ":" << std::endl;
                        //             std::cout << c1c4[i] << std::endl;
                        //         }
                            
                            cv::Mat l1= l1l4[0];
                            cv::Mat l2= l1l4[1];
                            cv::Mat l3= l1l4[2];
                            cv::Mat l4= l1l4[3];

                            cv::Mat c1=c1c4[0];
                            cv::Mat c2=c1c4[1];
                            cv::Mat c3=c1c4[2];
                            cv::Mat c4=c1c4[3];

                            // std::cout << l1l4[0].size() << std::endl;

                            float ms = get_score(l1,l2,l3,l4,c1,c2,c3,c4);
                            std::cout << ms<< std::endl;
                            if (ms < min_score) {
                                min_score = ms;
                                index = j;
                            }


                            if (min_score < svd_threshold) {
                                // std::cout << "-------------------------" << std::endl;
                                ids1[i][0] = index;
                                ids1[i][1] = -2;
                            }




                        }



                    }
                }
            }
        }






        // // Iterate through the map and print the key-value pairs
        // for (const auto& entry : ids1) {
        //     int key = entry.first;
        //     const std::vector<float>& values = entry.second;

        //     std::cout << "Key: " << key << ", Values: ";
            
        //     for (float value : values) {
        //         std::cout << value << " ";
        //     }

        //     std::cout << std::endl;
        // }    

        // // Iterate through the map and print the key-value pairs
        // for (const auto& entry : ids2) {
        //     int key = entry.first;
        //     const std::vector<float>& values = entry.second;

        //     std::cout << "Key: " << key << ", Values: ";
            
        //     for (float value : values) {
        //         std::cout << value << " ";
        //     }

        //     std::cout << std::endl;
        // }    



    }


    return result;

}



//end of not valid


int main(){

    // Define the rows and columns of your matrices
    int rows = 11;
    int cols = 6;

    // Create two empty matrices of the desired size and type (float)
    cv::Mat matrix1(rows, cols, CV_32F);
    

    // Assign the values to the first matrix
    float values1[11][6] = {
        {3.09823608e+02, 6.78893280e+01, 3.82249878e+02, 2.61026123e+02, 6.16126478e-01, 0.00000000e+00},
        {4.02368355e+00, 2.77120880e+02, 6.03552475e+01, 4.54162903e+02, 7.20909894e-01, 0.00000000e+00},
        {1.51692847e+03, 2.28836655e+02, 1.60544946e+03, 5.02447174e+02, 8.92754734e-01, 0.00000000e+00},
        {4.02368164e+00, 5.66826050e+02, 9.25447083e+01, 6.79489197e+02, 2.76628226e-01, 0.00000000e+00},
        {1.19905774e+03, 2.44931442e+02, 1.39219434e+03, 5.34636536e+02, 8.71798038e-01, 0.00000000e+00},
        {5.11007751e+02, 3.69665588e+02, 6.96097168e+02, 7.47891785e+02, 8.08927953e-01, 0.00000000e+00},
        {7.36333984e+02, 6.23157593e+02, 1.08237073e+03, 8.80673279e+02, 8.71798038e-01, 0.00000000e+00},
        {7.72547119e+02, 2.16765656e+02, 1.06225232e+03, 3.85760284e+02, 3.81411642e-01, 1.30000000e+01},
        {1.44852588e+03, 4.21973419e+02, 1.72213635e+03, 6.63394409e+02, 7.20909894e-01, 1.30000000e+01},
        {6.19647217e+02, 7.23749695e+02, 1.34391003e+03, 1.05369177e+03, 2.76628226e-01, 1.30000000e+01},
        {1.11456018e+03, 3.17357697e+02, 1.29964966e+03, 5.10494476e+02, 4.98769075e-01, 5.60000000e+01}
    };

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix1.at<float>(i, j) = values1[i][j];
        }
    }

    int rows2 = 13;
    int cols2 = 6;
    cv::Mat matrix2(rows2, cols2, CV_32F);

    // Assign the values to the second matrix
    float values2[13][6] = {
        {3.13847290e+02, 6.78893280e+01, 3.62131439e+02, 2.44931396e+02, 3.81411642e-01, 0.00000000e+00},
        {1.42438379e+03, 2.00670914e+02, 1.52095227e+03, 4.66234039e+02, 8.71798038e-01, 0.00000000e+00},
        {4.02368355e+00, 2.77120880e+02, 6.03552475e+01, 4.54162903e+02, 7.20909894e-01, 0.00000000e+00},
        {4.02368164e+00, 5.62802368e+02, 9.25447083e+01, 7.15702332e+02, 3.81411642e-01, 0.00000000e+00},
        {1.19905774e+03, 2.44931442e+02, 1.39219434e+03, 5.34636536e+02, 8.71798038e-01, 0.00000000e+00},
        {7.04144531e+02, 2.00670914e+02, 8.40949768e+02, 5.78897156e+02, 8.08927953e-01, 0.00000000e+00},
        {5.11007751e+02, 3.69665588e+02, 6.96097168e+02, 7.47891785e+02, 8.08927953e-01, 0.00000000e+00},
        {7.36333984e+02, 6.23157593e+02, 1.09846545e+03, 8.80673279e+02, 8.71798038e-01, 0.00000000e+00},
        {1.00592084e+03, 3.09310364e+02, 1.19905774e+03, 5.02447174e+02, 3.26924264e-01, 1.30000000e+01},
        {1.44852588e+03, 4.21973419e+02, 1.72213635e+03, 6.63394409e+02, 8.08927953e-01, 1.30000000e+01},
        {6.19647217e+02, 7.23749695e+02, 1.34391003e+03, 1.05369177e+03, 5.57447791e-01, 1.30000000e+01},
        {1.11456018e+03, 3.21381409e+02, 1.29964966e+03, 5.06470825e+02, 3.26924264e-01, 5.60000000e+01},
        {0.00000000e+00, 6.19133850e+02, 6.43789291e+01, 8.36412781e+02, 2.76628226e-01, 5.60000000e+01}
    };


    for (int i = 0; i < rows2; i++) {
        for (int j = 0; j < cols2; j++) {
            matrix2.at<float>(i, j) = values2[i][j];
        }
    }

    std::cout << "Matrix1:" << std::endl;
    std::cout << matrix1 << std::endl;

    std::cout << "Matrix2:" << std::endl;
    std::cout << matrix2 << std::endl;

    std::string img1="./test/test_case1/0030.jpg";
    std::string img2="./test/test_case1/0060.jpg";

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


    // std::map<int, std::vector<double>> outerMap = {
    //     {0, {0, 13.141539953319471, 0.0}},
    //     {1, {1, 9.967117874029203, 0.0}},
    //     // Add more entries as needed
    // };
    //then i think mine is vector<std::map<int, std::vector<double>>> 


    compare(
        img1, matrix1,output_id1,
        img2, matrix2,output_id2
    );

    return 0;
}


// g++ -I../tensorflow -ltensorflow_cc -c simple.cpp `pkg-config --cflags --libs opencv4`


//output data is (1,84,8400)


// results is 

