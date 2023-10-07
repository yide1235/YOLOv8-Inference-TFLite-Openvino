
!git clone https://gitee.com/ppov-nuc/yolov8_openvino.git

!pip install -r requirements.txt

pip install torch==2.0.1

pip install torchaudio==2.0.2+cu118 torchdata==0.6.1 torchtext==0.15.2

pip install torchvision --upgrade

!pip install ultralytics

!yolo export model=yolov8l-seg.pt format=onnx

cd yolov8_openvino

!mo -m yolov8l-seg.onnx --compress_to_fp16

!python export_yolov8_cls_ppp.py