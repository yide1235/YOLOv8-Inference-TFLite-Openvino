// Martin Kersner, m.kersner@gmail.com
// 2016/12/19 

#include "include/util/nms_utils.hpp"
using std::vector;

cv::Rect vec_to_rect(const vector<float> & vec)
{
  return cv::Rect(cv::Point(vec[0], vec[1]), cv::Point(vec[2], vec[3]));
}

void draw_rectangles(cv::Mat & img,
                    const vector<vector<float>> & vecVecFloat)
{
  for (const auto & vec: vecVecFloat)
    cv::rectangle(img, vec_to_rect(vec),  WHITE_COLOR);
}

void draw_rectangles(cv::Mat & img,
                    const vector<cv::Rect> & vecRect)
{
  for (const auto & rect: vecRect)
    cv::rectangle(img, rect,  WHITE_COLOR);
}
