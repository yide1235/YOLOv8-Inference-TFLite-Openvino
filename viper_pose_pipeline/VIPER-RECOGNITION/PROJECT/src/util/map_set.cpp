#include "../../include/util/map_set.h"


mutex yolov8_processing_mutex;
condition_variable yolov8_processing_start_cv;
mutex yolov8_post_process_mutex;
condition_variable yolov8_post_process_start_cv;
mutex person_database_management_mutex;
condition_variable person_database_management_start_cv;

