from ultralytics import YOLO
import cv2
import glob

# Load a model
model = YOLO('./yolov8n-seg.pt')  # load an official model
temp=cv2.imread('./test/download.png')
print(temp)


results = model('./test/download.png')  # predict on an image

print(results)
print(type(results))
print(results[0].mask)
# cv2.imshow(results)

# if __name__ == '__main__':
 
#         files = glob.glob(f'./test/*.png')


#         for file in files:
            
#             results = model(file)

#             for r in results:
#                 print(r.masks)


'''

pip install -U ultralytics

yolo export model=yolov8l.pt data=coco128.yaml format=tflite int8


LD_PRELOAD=/usr/lib/python3.8/lib-dynload/_ctypes.cpython-38-x86_64-linux-gnu.so yolo export model=yolov8l-seg.pt data=coco128-seg.yaml format=tflite

'''

