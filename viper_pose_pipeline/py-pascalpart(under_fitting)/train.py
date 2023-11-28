from ultralytics import YOLO
 
# Load the model.
model = YOLO('yolov8l-seg.pt')
 
# Training.
results = model.train(
   data='pascal-part-seg.yaml',
   imgsz=640,
   epochs=10,
   batch=16,
   name='yolov8l_seg_pascalpart')