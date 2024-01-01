#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>

void fillVector(std::vector<int>& vec, int size, int range) {
    std::random_device rd;
    std::mt19937 gen(rd());

    while(vec.size() < size) {
        vec.push_back(gen() % (range + 1));
    }

    // Sort the vector to use set_intersection
    std::sort(vec.begin(), vec.end());
}

int main() {
    const std::vector<int> sizes = {100, 1000, 10000,  100000, 1000000, 10000000};

    for (int size : sizes) {
        std::vector<int> vec1, vec2;
        fillVector(vec1, size, 50000);
        fillVector(vec2, size, 100000);

        std::vector<int> result;

        auto start = std::chrono::high_resolution_clock::now();

        // Use set_intersection from the standard library
        std::set_intersection(vec1.begin(), vec1.end(), vec2.begin(), vec2.end(), std::back_inserter(result));

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;

        std::cout << "Size: " << size << ", Time taken for comparison: " << elapsed.count() << " ms\n";
        vec1.clear();
        vec2.clear();
    }

    return 0;
}


// #include <iostream>
// #include <map>
// #include <vector>
// #include <chrono>
// #include <random>

// // Function to fill a vector with random numbers within a given range
// void fillVector(std::vector<int>& vec, int size, int range) {
//     std::random_device rd;
//     std::mt19937 gen(rd());

//     while(vec.size() < size) {
//         vec.push_back(gen() % (range + 1)); // Generate a random number within the range
//     }
// }

// int main() {
//     const std::vector<int> sizes = {100, 1000, 10000, 100000, 1000000, 10000000}; // Data sizes

//     for (int size : sizes) {
//         std::vector<int> vec1, vec2;
//         fillVector(vec1, size, 50000);
//         fillVector(vec2, size, 100000);

//         std::map<int, std::pair<int, int>> pairedMap; // Map to store paired elements

//         auto start = std::chrono::high_resolution_clock::now();

//         for (int i = 0; i < size; ++i) {
//             pairedMap[i] = std::make_pair(vec1[i], vec2[i]);
//         }

//         // Perform some operations on pairedMap if needed

//         auto end = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double, std::milli> elapsed = end - start;

//         std::cout << "Size: " << size << ", Time taken to create map: " << elapsed.count() << " ms\n";
//     }

//     return 0;
// }


//to run this: g++ -o test_unordered test_unordered.cpp -ltensorflow-lite -lopencv_core -lopencv_imgproc -lopencv_highgui `pkg-config --cflags --libs opencv4` && ./test_unordered
