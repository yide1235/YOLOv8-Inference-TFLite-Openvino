#include "../../include/threading/person_database_management_thread.hpp"
#include "../../include/util/map_set.h"
using namespace std;

std::unique_lock<std::mutex> person_database_management_lock(person_database_management_mutex);

bool keep_person_database_management_thread_alive(){
    
    person_database_management_lock.lock();
    bool keep_alive = true;
    person_database_management_lock.unlock();

    return keep_alive;
}

int binary_search(vector<float> arr, float value){
    int left=0;
    int right=arr.size()-1;
    int mid;
    if (right<0){
        return 0;
    }
    while (left<=right){
        mid=(left+right)/2;
        if (arr[mid]==value){
            return mid;
        }
        if (arr[mid]<value){
            left=mid+1;
        } else {
            right=mid-1;
        }
    }
    return mid;
}


struct result_vector_vec_vector_vector_vec {
    vector<int> vec;
    vector<vector<int>> vecvec;
};

result_vector_vec_vector_vector_vec filter_global_data(vector<vector<float>>arr,vector<float> lookup,float percentage){
    // arr in matrix
    // lookup is vector
    // percentage is float or int
    vector<int> index_matched;
    vector<vector<int>> list_inds;
    vector<int> empty_vec;
    float lower;
    float upper;
    std::vector<float> sub_arr;
    int lower_ind;
    int upper_ind;
    bool started=false;
    int last_0;
    result_vector_vec_vector_vector_vec result;

    for (int i=0;i<lookup.size();i++){
        if (lookup[i]==0){
            list_inds.push_back(empty_vec);
            continue;
        }

        lower = lookup[i]*(1.0-percentage);
        upper = lookup[i]*(1.0+percentage);
        
        for (int j = 0; i < arr.size(); j++) {
            sub_arr.push_back(arr[j][(i*2)+1]);
        }

        lower_ind = binary_search(sub_arr,lower);
        upper_ind = binary_search(sub_arr,upper);

        if (arr[0][i*2+1]){
            int j = 1;
            while (true){
                if (arr[j][i*2+1]!=0){
                    last_0=j;
                    break;
                }
                j=j+1;
            }
        } else {
            last_0=-1;
        }

        if (index_matched.size()==0 && started==false){
            for (int j = lower_ind; j <= upper_ind; ++j) {
                index_matched.push_back(arr[j][1 + i * 2]);
            }

            if (last_0 != -1) {
                for (int j = 0; j < last_0; ++j) {
                    index_matched.push_back(j);
                }
            }

            started = true;
            list_inds.push_back({lower_ind, upper_ind + 1, last_0});
        } else {
            std::vector<int> temp(index_matched);
            index_matched.clear();
            for (int j = lower_ind; j <= upper_ind; ++j) {
                if (std::find(temp.begin(), temp.end(), arr[j][1 + i * 2]) != temp.end()) {
                    index_matched.push_back(arr[j][1 + i * 2]);
                }
            }
            if (last_0 != -1) {
                for (int j = 0; j < last_0; ++j) {
                    index_matched.push_back(j);
                }
            }
            list_inds.push_back({lower_ind, upper_ind + 1, last_0});
        }
    }
    result.vec = index_matched;
    result.vecvec = list_inds;
    return result;
}



int person_database_management_thread()
{
    while(keep_person_database_management_thread_alive()){

        // DO STUFF
    }
}