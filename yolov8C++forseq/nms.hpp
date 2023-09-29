// Martin Kersner, m.kersner@gmail.com
// 2016/12/18

#ifndef NMS_HPP__
#define NMS_HPP__

#include <vector>
#include <numeric>
#include <opencv2/opencv.hpp>

enum PointInRectangle {XMIN, YMIN, XMAX, YMAX};

std::vector<cv::Rect> nms_vec(const std::vector<std::vector<float>> &,
                          const float &, std::vector<int> &);

std::vector<float> get_point_from_rect(const std::vector<std::vector<float>> &,
                                    const PointInRectangle &);

std::vector<float> compute_area(const std::vector<float> &,
                               const std::vector<float> &,
                               const std::vector<float> &,
                               const std::vector<float> &);

template <typename T>
std::vector<int> argsort(const std::vector<T> & v);

std::vector<float> nms_maximum(const float &,
                           const std::vector<float> &);

std::vector<float> nms_minimum(const float &,
                           const std::vector<float> &);

std::vector<float> copy_by_indexes(const std::vector<float> &,
                                 const std::vector<int> &);

std::vector<int> remove_last(const std::vector<int> &);

std::vector<float> subtract(const std::vector<float> &,
                            const std::vector<float> &);

std::vector<float> multiply(const std::vector<float> &,
                            const std::vector<float> &);

std::vector<float> divide(const std::vector<float> &,
                          const std::vector<float> &);

std::vector<int> where_larger(const std::vector<float> &,
                             const float &);

std::vector<int> remove_by_indexes(const std::vector<int> &,
                                 const std::vector<int> &);

std::vector<cv::Rect> boxes_to_rectangles(const std::vector<std::vector<float>> &);

template <typename T>
std::vector<T> filter_vector(const std::vector<T> &,
                            const std::vector<int> &);

#endif // NMS_HPP__
