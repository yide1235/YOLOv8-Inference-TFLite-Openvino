import argparse
import time
import pdb
import os
import glob
import numpy as np
import cv2 as cv
import tflite_runtime.interpreter as tflite
from tqdm import tqdm


coco_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
              'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 
              'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 
              'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
              'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 
              'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
              'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
              'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 
              'teddy bear', 'hair drier', 'toothbrush']

model_name = 'yolov8l_integer_quant'

class YOLOV8:
    def __init__(self) -> None:
        self.interpreter = tflite.Interpreter(model_path=f'./{model_name}.tflite')
        # self.interpreter = tflite.Interpreter(model_path='models/yolov8n_int8.tflite')
        # self.interpreter = tflite.Interpreter(model_path='models/yolov8l_int8.tflite',
        #                 experimental_delegates=[tflite.load_delegate('vx_delegate.so')])
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # check the type of the input tensor
        self.floating_model = self.input_details[0]['dtype'] == np.float32

        # NxHxWxC, H:1, W:2
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]

        self.img_height = 0
        self.img_width = 0

        # parameters
        self.conf_thres = 0.25
        self.overlapThresh = 0.45

    def preprocess(self, image):
        # load image
        if type(image) == str:  # Load from file path
            if not os.path.isfile(image):
                raise ValueError("Input image file path (", image, ") does not exist.")
            image = cv.imread(image)
        elif isinstance(image, np.ndarray):  # Use given NumPy array
            image = image.copy()
        else:
            raise ValueError("Invalid image input. Only file paths or a NumPy array accepted.")
        
        self.img_height = image.shape[0]
        self.img_width = image.shape[1]

        # resize and padding
        image = self.letterbox(image)
        
        # BGR -> RGB
        image = image[:, :, ::-1]
        # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # add N dim
        input_data = np.expand_dims(image, axis=0)
        
        if self.floating_model:
            input_data = np.float32(input_data) / 255
        else:
            input_data = input_data.astype(np.int8)

        return input_data

    def detect(self, image, object=None):
        interpreter = self.interpreter

        input_data = self.preprocess(image)

        interpreter.set_tensor(self.input_details[0]['index'], input_data)
        # start_time = time.time()
        interpreter.invoke()
        # stop_time = time.time()
        output_data = interpreter.get_tensor(self.output_details[0]['index'])
        
        results = self.postprocess(output_data)

        if object is not None:  
            results = [result for result in results if result['cls_name'] == object]

        return results
    
    def postprocess(self, output_data):
        output = np.squeeze(output_data).T
        boxes, probs = output[:, :4], output[:, 4:]

        # select high confident bboxes
        scores = np.amax(probs, axis=1)
        idx_th = np.where(scores > self.conf_thres)[0]

        boxes = boxes[idx_th]
        scores = scores[idx_th]

        preds = np.argmax(probs[idx_th], axis=1)

        boxes = self.xywh2xyxy_scale(boxes)
        n_classes = probs.shape[1]

        results = []
        for i in range(n_classes):
            # select bboxes with class i
            ind = np.where(preds==i)[0]
            if len(ind) > 0:
                indices = self.NMS(boxes[ind])

                box = boxes[ind[indices]]
                box = self.scale_boxes(box)

                score = scores[ind[indices]]

                for bbox, sscore in zip(box, score):
                    result = {
                        'bounding_box': bbox,
                        'cls_id': i,
                        'cls_name': coco_names[i],
                        'score': sscore
                    }
                    results.append(result)

        return results
    
    def letterbox(self, img):
        """Resize image and pad to square"""
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = (self.height, self.width)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv.resize(img, new_unpad, interpolation=cv.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT,
                                value=(114, 114, 114))  # add border
        
        return img
    
    def xywh2xyxy_scale(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = (x[:, 0] - x[:, 2] / 2) * self.width  # top left x
        y[:, 1] = (x[:, 1] - x[:, 3] / 2) * self.height  # top left y
        y[:, 2] = (x[:, 0] + x[:, 2] / 2) * self.width  # bottom right x
        y[:, 3] = (x[:, 1] + x[:, 3] / 2) * self.height  # bottom right y
        return y
    
    def NMS(self, boxes):
        # Return an empty list, if no boxes given
        if len(boxes) == 0:
            return []
        x1 = boxes[:, 0]  # x coordinate of the top-left corner
        y1 = boxes[:, 1]  # y coordinate of the top-left corner
        x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
        y2 = boxes[:, 3]  # y coordinate of the bottom-right corner
        # Compute the area of the bounding boxes and sort the bounding
        # Boxes by the bottom-right y-coordinate of the bounding box
        areas = (x2 - x1 + 1) * (y2 - y1 + 1) # We add 1, because the pixel at the start as well as at the end counts
        # The indices of all boxes at start. We will redundant indices one by one.
        indices = np.arange(len(x1))
        for i, box in enumerate(boxes):
            # Create temporary indices  
            temp_indices = indices[indices!=i]
            # Find out the coordinates of the intersection box
            xx1 = np.maximum(box[0], boxes[temp_indices,0])
            yy1 = np.maximum(box[1], boxes[temp_indices,1])
            xx2 = np.minimum(box[2], boxes[temp_indices,2])
            yy2 = np.minimum(box[3], boxes[temp_indices,3])
            # Find out the width and the height of the intersection box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            # compute the ratio of overlap
            overlap = (w * h) / areas[temp_indices]
            # if the actual bounding box has an overlap bigger than treshold with any other box, remove it's index  
            if np.any(overlap) > self.overlapThresh:
                indices = indices[indices != i]
        #return only the boxes at the remaining indices
        return indices

    def scale_boxes(self, boxes):
        """
        Rescales bounding boxes (in the format of xyxy) from the shape of the image they were originally specified in
        (img1_shape) to the shape of a different image (img0_shape).
        """
        img1_shape = (self.height, self.width)
        img0_shape = (self.img_height, self.img_width)
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1), round(
            (img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)  # wh padding

        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
        boxes[..., :4] /= gain

        # clip boxes
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, img0_shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, img0_shape[0])  # y1, y2

        return boxes


class BboxesPlotter:
    def __init__(self) -> None:
        self.colors = self.Colors()

    class Colors:
        # Ultralytics color palette https://ultralytics.com/
        def __init__(self):
            # hex = matplotlib.colors.TABLEAU_COLORS.values()
            hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
            self.palette = [self.hex2rgb('#' + c) for c in hex]
            self.n = len(self.palette)

        def __call__(self, i, bgr=False):
            c = self.palette[int(i) % self.n]
            return (c[2], c[1], c[0]) if bgr else c

        @staticmethod
        def hex2rgb(h):  # rgb order (PIL)
            return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    def plot_one_box(self, x, im, color=(128, 128, 128), label=None, line_thickness=3):
        # Plots one bounding box on image 'im' using OpenCV
        tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv.rectangle(im, c1, c2, color, thickness=tl, lineType=cv.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv.rectangle(im, c1, c2, color, -1, cv.LINE_AA)  # filled
            cv.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv.LINE_AA)
        
        return im

    def plot_bboxes(self, img_path, results, save_path):
        
        im0 = cv.imread(img_path)
        for obj in results:
            bbox = obj['bounding_box']
            cls_id = obj['cls_id']
            cls_name = obj['cls_name']
            score = obj['score']
            label = f'{cls_name} {score:.2f}'
            color = self.colors(cls_id, True)

            im0 = self.plot_one_box(bbox, im0, color, label)

        cv.imwrite(save_path, im0)


if __name__ == '__main__':

        file = './input.png'

        yolo = YOLOV8()
        plotter = BboxesPlotter()


            
        start = time.time()
        results = yolo.detect(file)
        print(f'time: {time.time() - start} s')

        save_name = './output.png'
        dir_name = os.path.dirname(save_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        
        plotter.plot_bboxes(file, results, save_name)
