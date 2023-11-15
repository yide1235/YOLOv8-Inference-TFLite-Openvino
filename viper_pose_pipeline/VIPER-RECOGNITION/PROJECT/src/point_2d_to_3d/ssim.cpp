#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>


cv::Mat pad_image(const cv::Mat& image, int padding) {
    // takes in image and pad size and returns image with outside border of zeros
    int padded_height = image.rows + 2 * padding;
    int padded_width = image.cols + 2 * padding;
    cv::Mat padded_image = cv::Mat::zeros(padded_height, padded_width, image.type());
    image.copyTo(padded_image(cv::Rect(padding, padding, image.cols, image.rows)));
    return padded_image;
}

cv::Mat smoother(const cv::Mat& image, int filter_size) {
    // takes in image and filter_size to compute uniform filter on the image (similar to blurring)
    int height = image.rows;
    int width = image.cols;
    int half_filter_size = filter_size / 2;

    cv::Mat padded_image = pad_image(image, half_filter_size);
    cv::Mat kernel = cv::Mat::ones(filter_size, filter_size, CV_32F) / static_cast<float>(filter_size * filter_size);
    cv::Mat blurred_image = cv::Mat::zeros(image.size().width,image.size().height , CV_32FC1);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            cv::Rect roi(x, y, filter_size, filter_size);
            cv::Mat neighborhood = padded_image(roi);
            float blurred_value = cv::sum(neighborhood.mul(kernel))[0];
            blurred_image.at<float>(y, x) = blurred_value;
        }
    }

    return blurred_image;
}

float ssim(const cv::Mat& im1, const cv::Mat& im2, int channel_axis, bool multichannel, int win_size) {
    // tames in two images (im1,im2) the index for the colour channel (always 2 when using opencv), boolean for multichannel (used for recusion, will always be called with true on run), win_size used in smoother (see above)
    cv::Mat im1_float, im2_float;
    im1.convertTo(im1_float, CV_32F);
    im2.convertTo(im2_float, CV_32F);

    if (multichannel) {
        std::vector<cv::Mat> im1_channels, im2_channels;
        cv::split(im1_float, im1_channels);
        cv::split(im2_float, im2_channels);

        std::vector<float> mssims;
        for (int i = 0; i < im1.channels(); ++i) {
            cv::Mat im1_single_channel = im1_channels[i];
            cv::Mat im2_single_channel = im2_channels[i];
            mssims.push_back(ssim(im1_single_channel, im2_single_channel, -1, false, win_size));
        }
        float mean_mssim = std::accumulate(mssims.begin(), mssims.end(), 0.0f) / mssims.size();
        return mean_mssim;
    } else {
        cv::Mat ux = smoother(im1_float, win_size);
        cv::Mat uy = smoother(im2_float, win_size);
        cv::Mat uxx = smoother(im1_float.mul(im1_float), win_size);
        cv::Mat uxy = smoother(im1_float.mul(im2_float), win_size);
        cv::Mat uyy = smoother(im2_float.mul(im2_float), win_size);
        int NP = win_size * win_size;
        float cov_norm = static_cast<float>(NP) / (NP - 1);
        cv::Mat uxux = ux.mul(ux);
        cv::Mat uxuy = ux.mul(uy);
        cv::Mat uyuy = uy.mul(uy);
        cv::Mat vx = cov_norm * (uxx - uxux);
        cv::Mat vy = cov_norm * (uyy - uyuy);
        cv::Mat vxy = cov_norm * (uxy - uxuy);
        float k1 = 0.01f;
        float k2 = 0.03f;
        float R = 2.0f;
        float c1 = std::pow(k1 * R, 2);
        float c2 = std::pow(k2 * R, 2);
        cv::Mat A1 = 2.0f * uxuy + c1;
        cv::Mat A2 = 2.0f * vxy + c2;
        cv::Mat B1 = uxux + uyuy + c1;
        cv::Mat B2 = vx + vy + c2;
        cv::Mat ssim_mat = A1.mul(A2) / (B1.mul(B2));
        int pad = (win_size - 1) / 2;
        cv::Rect roi(pad, pad, im1.cols - pad * 2, im1.rows - pad * 2);
        cv::Mat ssim_region = ssim_mat(roi);
        float mean_mssim = cv::mean(ssim_region)[0];
        return mean_mssim;
    }
}



// int main(int argc, char* argv[]) {
//     cv::Mat imgl = cv::imread("../../Dropbox/REALTIME7/tmpl.jpg", cv::IMREAD_COLOR);
//     cv::Mat imgr = cv::imread("../../Dropbox/REALTIME7/tmpr.jpg", cv::IMREAD_COLOR);

//     cv::Mat image1, image2;


//     image1=imgl;
//     image2=imgr;

//     cv::Point p1(100, 1000);
//     int target_col = 1090;
//     int box_radius = std::stoi(argv[1]);
//     cv::Rect roi1(p1.x - box_radius, p1.y - box_radius, 2*box_radius+1, 2*box_radius+1);
//     cv::Mat im1 = image1(roi1);
    
//     std::vector<float> ssims;
//     cv::Rect roi2(1040 - box_radius, p1.y - box_radius, 2*box_radius+1, 2*box_radius+1);
//     cv::Mat im2 = image2(roi2);

    
//     int channel_axis = 2;
//     bool multichannel = true;
//     int win_size = 7;

//     double start = static_cast<double>(cv::getTickCount());
//     float similarity = ssim(im1, im2, channel_axis, multichannel, win_size);
//     double elapsed_time = static_cast<double>(cv::getTickCount()) - start;

//     std::cout << "SSIM: " << similarity << std::endl;
    
//     std::cout << "Time taken: " << elapsed_time / cv::getTickFrequency() << " seconds" << std::endl;

//     return 0;
// }
