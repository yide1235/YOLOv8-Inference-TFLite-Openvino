# yolov8-object-tracking 
#### [ultralytics==8.0.0]


### Features
- Object Tracks
- Different Color for every track
- Video/Image/WebCam/External Camera/IP Stream Supported

### Coming Soon
- Selection of specific class ID for tracking
- Development of dashboard for YOLOv8

### Train YOLOv8 on Custom Data
- https://chr043416.medium.com/train-yolov8-on-custom-data-6d28cd348262

### Steps to run Code

- Clone the repository
```
https://github.com/yide1235/Yolov8-tracking-tflite-CPP.git
```

- Goto cloned folder
```
cd yolov8-object-tracking
```

- Install the ultralytics package
```
pip install ultralytics==8.0.0
```

- Do Tracking with mentioned command below
```
#video file
python ./detect_and_trk.py model=yolov8s.pt source="test.mp4" show=True

#imagefile
python ./detect_and_trk.py model=yolov8m.pt source="path to image"

#Webcam
python ./detect_and_trk.py model=yolov8m.pt source=0 show=True

#External Camera
python ./detect_and_trk.py model=yolov8m.pt source=1 show=True
```

- Output file will be created in the ./runs/detect/train with original filename


### Results
<table>
  <tr>
    <td>YOLOv8s Object Tracking</td>
    <td>YOLOv8m Object Tracking</td>
  </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/62513924/211671576-7d39829a-f8f5-4e25-b30a-530548c11a24.png"></td>
    <td><img src="https://user-images.githubusercontent.com/62513924/211672010-7415ef8b-7941-4545-8434-377d94675299.png"></td>
  </tr>
 </table>

### References
- https://github.com/abewley/sort
- https://github.com/ultralytics/ultralytics
- https://github.com/RizwanMunawar/yolov8-object-tracking.git
