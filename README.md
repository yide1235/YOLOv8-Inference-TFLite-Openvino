# YOLOv8-Inference-TFLite-Openvino

#### Implementation of Yolov8 detection inference, segmentation inference, tracking for stereo image pair,in TFLite and openvino

-------
### Steps to run Code

- Clone the repository
```
https://github.com/yide1235/Yolov8-tracking-tflite-CPP.git
```
### Results
Tracking on objects on image pairs(YOLOv8-TFLite-object-matching-python,YOLOv8-TFLite-object-matching-C++)
![](./assets/1.jpg)
![](./assets/2.jpg)

YOLOv8l tflite detection result:
![](./assets/3.jpg)

YOLOv8l tflite segmentation result:
![](./assets/4.jpg)

-------

### C++run command figure out:
//help running command
//my part dont need tflite
//$ g++ -I../tensorflow -ltensorflow_cc -c test.cpp `pkg-config --cflags --libs opencv4`
//$ g++ -I../tensorflow -ltensorflow_cc -c test.o `pkg-config --cflags --libs 
opencv4`

-------


//using original yolo repo for inference:

(some env mayhelp: pip install torch==2.0.1
pip install tensorflow==2.13.1 #2.14 will give error

pip install torchaudio==2.0.2+cu118 torchdata==0.6.1 torchtext==0.15.2

pip install torchvision --upgrade

!pip install ultralytics)

yolo export model=yolov8l-seg.pt data=coco128-seg.yaml format=tflite int8

yolo predict model=./yolov8x-seg_int8.tflite source='./download2.png'

-------

Now train it for human part segmentation
Somecode: from ultralytics import YOLO

model_name = 'yolov8l-seg' #@param ["yolov8n-seg", "yolov8s-seg", "yolov8m-seg", "yolov8l-seg", "yolov8x-seg"]
input_width = 640 #@param {type:"slider", min:32, max:4096, step:32}
input_height = 640 #@param {type:"slider", min:32, max:4096, step:32}
optimize_cpu = False

model = YOLO(f"{model_name}.pt") 
model.export(format="tflite", imgsz=[input_height,input_width], optimize=optimize_cpu, int8=True)

//current converts pascal-part to yolo format

//doing the training today

//connect to clearml: !pip install clearml
!pip install clearml>=1.2.0
!clearml-init


//training cli: !yolo task=segment mode=train model=yolov8l-seg.pt data=pascal-part-seg.yaml epochs=300 batch=8

//resume training cli example: !yolo task=segment mode=train resume model=./runs/segment/train6/weights/last.pt data=pascal-part-seg.yaml epochs=200 batch=12 

//new a .py file called train.py and this to retrain:
from ultralytics import YOLO

if __name__ == '__main__':
  
  model = YOLO('./last.pt') # I copied last.pt to the principal folder
  model.resume=True

  results = model.train(
    data='./pascal-part-seg.yaml',
    imgsz=640,
    epochs=500,
    batch=12)

-------

###right now training the first 300 epoch, then last 300 epoch
### testing
source yolov8python/bin/activate
yolo predict model=./last.pt source='./tmpr21.png' hide_labels=True boxes=False

-------


### To check how to install tflite on C++, like rebuild tflite with C++, check here:
https://github.com/karthickai/tflite.git

### for code from yolov8C++_motion_detection: function like this should be modified:
void generateIds(std::vector<std::vector<float>>* results) {
    for (int i = 0; i < (*results).size(); ++i) {
      (*results)[i].push_back(0.0);
      (*results)[i].push_back(static_cast<float>(i));
      (*results)[i].push_back(-1.0);
      (*results)[i].push_back(0.0);
    }
    
}
//so instead of using (*results), just use results, so the modified version:
void generateIds(std::vector<std::vector<float>>& results) {
    for (int i = 0; i < (results).size(); ++i) {
      (results)[i].push_back(0.0);
      (results)[i].push_back(static_cast<float>(i));
      (results)[i].push_back(-1.0);
      (results)[i].push_back(0.0);
    }
    
}

//this make cause error, but it is fixed at yolov8C++_ssim

# end of results

### References
- https://github.com/abewley/sort
- https://github.com/ultralytics/ultralytics
- https://github.com/RizwanMunawar/yolov8-object-tracking.git





