# yolov8-object-tracking
#### Implementation of Yolov8l tracking with tflite in C++


### Phase:
//first experiments tracking algorithm in python
//then converts it in C++
//integerated with video and motion detection
//do frame1, frame2, then use frame2 id to do frame2 and frame3
//so for ssim, it is the matching algorithm between two images, may need to tune parameter when use
//for seq, just got a new one called yolov8forseq

### Features
- Object Tracks
- Different Color for every track
- Video/Image/WebCam/External Camera/IP Stream Supported

### Coming Soon
- Selection of specific class ID for tracking
- Development of dashboard for YOLOv8

### Steps to run Code

- Clone the repository
```
https://github.com/yide1235/Yolov8-tracking-tflite-CPP.git
```

### Results
Initial result using yolov8l quantized tflite pretrain model for stereo images(the first index is tracking id, the second is confidence, the color of the box is the class, you can notice this is two frame from a sequence):
![](./assets/1.jpg)
![](./assets/2.jpg)


### References
- https://github.com/abewley/sort
- https://github.com/ultralytics/ultralytics
- https://github.com/RizwanMunawar/yolov8-object-tracking.git



### C++run command figure out:
//help running command
//my part dont need tflite
//$ g++ -I../tensorflow -ltensorflow_cc -c test.cpp `pkg-config --cflags --libs opencv4`


//$ g++ -I../tensorflow -ltensorflow_cc -c test.o `pkg-config --cflags --libs 
opencv4`

