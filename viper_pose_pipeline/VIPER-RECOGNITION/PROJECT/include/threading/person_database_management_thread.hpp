#ifndef person_database_management_hpp
#define person_database_management_hpp

#include <vector>
#include <iostream>
#include <chrono>

class FrameData {
public:
    int id;
    vector<double> data;
    vector<double> bounding_box;
    vector<double> face_box;
    double face_score;
    chrono::system_clock::time_point daytime;

    FrameData(int frame_id, vector<double> data, vector<double> bounding_box, vector<double> face_box, double face_score, chrono::system_clock::time_point daytime)
        : id(frame_id),
          data(data),
          bounding_box(bounding_box),
          face_box(face_box),
          face_score(face_score),
          daytime(daytime) {
    }
};

class GlobalPerson {
public:
    int count;
    int global_id;
    std::vector<int> frame_ids;
    vector<double> face_frame;
    vector<double> bounding_box;
    vector<double> face_box;
    double face_score;

    GlobalPerson(const Person& person)
        : count(1),
          global_id(person.global_id),
          frame_ids(person.frame_ids),
          face_frame(person.face_frame),
          bounding_box(person.bounding_box),
          face_box(person.face_box),
          face_score(person.face_score) {
    }
};




#endif person_database_management_hpp