// Martin Kersner, m.kersner@gmail.com
// 2016/12/18
// C++ version of http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

#include "nms.hpp"
using std::vector;
using cv::Rect;
using cv::Point;

vector<Rect> nms_vec(const vector<vector<float>> & boxes,
                 const float & threshold, vector<int> & pick)
{
  if (boxes.empty())
  	return vector<Rect>();
  
  // grab the coordinates of the bounding boxes
  auto x1 = get_point_from_rect(boxes, XMIN);
  auto y1 = get_point_from_rect(boxes, YMIN);
  auto x2 = get_point_from_rect(boxes, XMAX);
  auto y2 = get_point_from_rect(boxes, YMAX);
  
  // compute the area of the bounding boxes and sort the bounding
  // boxes by the bottom-right y-coordinate of the bounding box
  auto area = compute_area(x1, y1, x2, y2);
  auto idxs = argsort(y2);
  
  int last;
  int i;
  //vector<int> pick;
  
  // keep looping while some indexes still remain in the indexes list
  while (idxs.size() > 0) {
    // grab the last index in the indexes list and add the
    // index value to the list of picked indexes
    last = idxs.size() - 1;	
    i    = idxs[last];
    pick.push_back(i);
    
    // find the largest (x, y) coordinates for the start of
    // the bounding box and the smallest (x, y) coordinates
    // for the end of the bounding box
    auto idxsWoLast = remove_last(idxs);

    auto xx1 = nms_maximum(x1[i], copy_by_indexes(x1, idxsWoLast));
    auto yy1 = nms_maximum(y1[i], copy_by_indexes(y1, idxsWoLast));
    auto xx2 = nms_minimum(x2[i], copy_by_indexes(x2, idxsWoLast));
    auto yy2 = nms_minimum(y2[i], copy_by_indexes(y2, idxsWoLast));

		// compute the width and height of the bounding box
    auto w = nms_maximum(0, subtract(xx2, xx1));
    auto h = nms_maximum(0, subtract(yy2, yy1));
		
		// compute the ratio of overlap
    auto overlap = divide(multiply(w, h), copy_by_indexes(area, idxsWoLast));

    // delete all indexes from the index list that have
    auto deleteIdxs = where_larger(overlap, threshold);
    deleteIdxs.push_back(last);
    idxs = remove_by_indexes(idxs, deleteIdxs);
  }

  return boxes_to_rectangles(filter_vector(boxes, pick));
}

vector<float> get_point_from_rect(const vector<vector<float>> & rect,
                               const PointInRectangle & pos)
{
  vector<float> points;
  
  for (const auto & p: rect)
    points.push_back(p[pos]);
  
  return points;
}

vector<float> compute_area(const vector<float> & x1,
                          const vector<float> & y1,
                          const vector<float> & x2,
                          const vector<float> & y2)
{
  vector<float> area;
  auto len = x1.size();
  
  for (decltype(len) idx = 0; idx < len; ++idx) {
    auto tmpArea = (x2[idx] - x1[idx] + 1) * (y2[idx] - y1[idx] + 1);
    area.push_back(tmpArea);
  }
  
  return area;
}

template <typename T>
vector<int> argsort(const vector<T> & v)
{
  // initialize original index locations
  vector<int> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);
  
  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(),
       [&v](int i1, int i2) {return v[i1] < v[i2];});
  
  return idx;
}

vector<float> nms_maximum(const float & num,
                      const vector<float> & vec)
{
  auto maxVec = vec;
  auto len = vec.size();
  
  for (decltype(len) idx = 0; idx < len; ++idx)
    if (vec[idx] < num)
      maxVec[idx] = num;
  
  return maxVec;
}

vector<float> nms_minimum(const float & num,
                      const vector<float> & vec)
{
  auto minVec = vec;
  auto len = vec.size();
  
  for (decltype(len) idx = 0; idx < len; ++idx)
    if (vec[idx] > num)
      minVec[idx] = num;
  
  return minVec;
}

vector<float> copy_by_indexes(const vector<float> & vec,
                            const vector<int> & idxs)
{
  vector<float> resultVec;
  
  for (const auto & idx : idxs)
    resultVec.push_back(vec[idx]);
  
  return resultVec;
}

vector<int> remove_last(const vector<int> & vec)
{
  auto resultVec = vec;
  resultVec.erase(resultVec.end()-1);
  return resultVec;
}

vector<float> subtract(const vector<float> & vec1,
                       const vector<float> & vec2)
{
  vector<float> result;
  auto len = vec1.size();
  
  for (decltype(len) idx = 0; idx < len; ++idx)
    result.push_back(vec1[idx] - vec2[idx] + 1);
  
  return result;
}

vector<float> multiply(const vector<float> & vec1,
		                   const vector<float> & vec2)
{
  vector<float> resultVec;
  auto len = vec1.size();
  
  for (decltype(len) idx = 0; idx < len; ++idx)
    resultVec.push_back(vec1[idx] * vec2[idx]);
  
  return resultVec;
}

vector<float> divide(const vector<float> & vec1,
		                 const vector<float> & vec2)
{
  vector<float> resultVec;
  auto len = vec1.size();
  
  for (decltype(len) idx = 0; idx < len; ++idx)
    resultVec.push_back(vec1[idx] / vec2[idx]);
  
  return resultVec;
}

vector<int> where_larger(const vector<float> & vec,
                        const float & threshold)
{
  vector<int> resultVec;
  auto len = vec.size();
  
  for (decltype(len) idx = 0; idx < len; ++idx)
    if (vec[idx] > threshold)
      resultVec.push_back(idx);
  
  return resultVec;
}

vector<int> remove_by_indexes(const vector<int> & vec,
                            const vector<int> & idxs)
{
  auto resultVec = vec;
  auto offset = 0;
  
  for (const auto & idx : idxs) {
    resultVec.erase(resultVec.begin() + idx + offset);
    offset -= 1;
  }
  
  return resultVec;
}

vector<Rect> boxes_to_rectangles(const vector<vector<float>> & boxes)
{
  vector<Rect> rectangles;
  vector<float> box;
  
  for (const auto & box: boxes)
    rectangles.push_back(Rect(Point(box[0], box[1]), Point(box[2], box[3])));
  
  return rectangles;
}

template <typename T>
vector<T> filter_vector(const vector<T> & vec,
    const vector<int> & idxs)
{
  vector<T> resultVec;
  
  for (const auto & idx: idxs){
    std::cout << " IDX is " << idx << std::endl;
    const auto box = vec[idx];
    std::cout << " Rect is " << box[0] << " " << box[1] << " " << box[2] << " " << box[3] << " " << std::endl;
    resultVec.push_back(vec[idx]); 
  }
  
  return resultVec;
}
