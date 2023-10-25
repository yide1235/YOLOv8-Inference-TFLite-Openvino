import math
import time
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import os
import glob

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']
               
       

# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)
# colors = rng.uniform(0, 255, size=(len(class_names), 3))

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

    # def __getitem__(self, index):
    #     return self.palette[index]

    def __getitem__(self, index):
        return self.palette[index % self.n]

    def __len__(self):
        return len(self.palette)

colors = Colors()
# print(type(colors))


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    np.seterr(divide='ignore', invalid='ignore')
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3, mask_maps=None, draw_bbox=False):

    img_height, img_width = image.shape[:2]
    size = min([img_height, img_width]) * 0.0010
    text_thickness = int(min([img_height, img_width]) * 0.002)

    mask_img = draw_masks(image, boxes, class_ids, mask_alpha, mask_maps, bg_removal=True)

    # Draw bounding boxes and labels of detections
    for box, score, class_id in zip(boxes, scores, class_ids):

        color = colors[class_id]
        
        color=colors(class_id, True)

        x1, y1, x2, y2 = box.astype(int)

        #-----------------------comment this part if you dont want the bbox
        # Draw rectangle
        if draw_bbox:
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, 2)
        #-----------------------end of this part

        label = class_names[class_id]
        caption = f'{label} {int(score * 100)}%'
        (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=size, thickness=text_thickness)
        th = int(th * 1.2)


        #-----------------------comment this part if you dont want the bbox
        if draw_bbox:
            cv2.rectangle(mask_img, (x1, y1),
                        (x1 + tw, y1 - th), color, -1)

            cv2.putText(mask_img, caption, (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)
        #-----------------------end of this part

    return mask_img



#this is normal printing
def draw_masks(image, boxes, class_ids, mask_alpha=0.3, mask_maps=None, bg_removal=None):
    
    if bg_removal:
        # Create a full white image with the same size as the input image
        mask_img = np.ones_like(image) * 255

        # Draw bounding boxes and labels of detections
        for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
            # color = colors[class_id]
            color = colors(class_id, True)

            x1, y1, x2, y2 = box.astype(int)

            if mask_maps is None:
                mask_img[y1:y2, x1:x2] = image[y1:y2, x1:x2]
            else:
                crop_mask = mask_maps[i][y1:y2, x1:x2, np.newaxis]
                mask_img[y1:y2, x1:x2] = image[y1:y2, x1:x2] * crop_mask + mask_img[y1:y2, x1:x2] * (1 - crop_mask)

        return mask_img
    
    else:
        mask_img = image.copy()

        # Draw bounding boxes and labels of detections
        for i, (box, class_id) in enumerate(zip(boxes, class_ids)):

            # color = colors[class_id]
            color=colors(class_id, True)

            x1, y1, x2, y2 = box.astype(int)


            if mask_maps is None:
                cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
            else:
                crop_mask = mask_maps[i][y1:y2, x1:x2, np.newaxis]
                crop_mask_img = mask_img[y1:y2, x1:x2]
                crop_mask_img = crop_mask_img * (1 - crop_mask) + crop_mask * color
                mask_img[y1:y2, x1:x2] = crop_mask_img

        return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)



def draw_comparison(img1, img2, name1, name2, fontsize=2.6, text_thickness=3):
    (tw, th), _ = cv2.getTextSize(text=name1, fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                  fontScale=fontsize, thickness=text_thickness)
    x1 = img1.shape[1] // 3
    y1 = th
    offset = th // 5
    cv2.rectangle(img1, (x1 - offset * 2, y1 + offset),
                  (x1 + tw + offset * 2, y1 - th - offset), (0, 115, 255), -1)
    cv2.putText(img1, name1,
                (x1, y1),
                cv2.FONT_HERSHEY_DUPLEX, fontsize,
                (255, 255, 255), text_thickness)

    (tw, th), _ = cv2.getTextSize(text=name2, fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                  fontScale=fontsize, thickness=text_thickness)
    x1 = img2.shape[1] // 3
    y1 = th
    offset = th // 5
    cv2.rectangle(img2, (x1 - offset * 2, y1 + offset),
                  (x1 + tw + offset * 2, y1 - th - offset), (94, 23, 235), -1)

    cv2.putText(img2, name2,
                (x1, y1),
                cv2.FONT_HERSHEY_DUPLEX, fontsize,
                (255, 255, 255), text_thickness)

    combined_img = cv2.hconcat([img1, img2])
    if combined_img.shape[1] > 3840:
        combined_img = cv2.resize(combined_img, (3840, 2160))

    return combined_img



model_name = 'yolov8x-seg_integer_quant'
# model_name = 'yolov8l-seg_integer_quant'
# model_name='last_quant_integer_quant'
# model_name= 'last_quant_int8'
# model_name = 'yolov8x-seg_int8'
# model_name='yolov8l-seg_float32'

# Define the YOLOSeg
class YOLOSeg_bg:

    def __init__(self):
        self.interpreter = tflite.Interpreter(model_path=f'./{model_name}.tflite')

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

        self.input_height = 640
        self.input_width = 640

        # parameters
        self.conf_threshold = 0.33
        self.iou_threshold = 0.20

        self.num_masks = 32



    def __call__(self, image):
        return self.segment_objects(image)



    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        # input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor




    def segment_objects(self, image):
        interpreter = self.interpreter

        input_data=self.prepare_input(image)
        # print(input_data.shape)
        # Perform inference on the image
        interpreter.set_tensor(self.input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data1 = interpreter.get_tensor(self.output_details[0]['index'])
        output_data2 = interpreter.get_tensor(self.output_details[1]['index'])


        self.boxes, self.scores, self.class_ids, mask_pred = self.process_box_output(output_data1)
        

        self.mask_maps = self.process_mask_output(mask_pred, output_data2)

        return self.boxes, self.scores, self.class_ids, self.mask_maps



    def expand_bounding_boxes(self, boxes):

        expanded_boxes = []
        add=0.30
        for box in boxes:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            expanded_x1 = x1 - add * width
            if(expanded_x1>0):
                x1=expanded_x1
            else:
                x1=0

            expanded_x2 = x2 + add * width
            if(expanded_x2<self.img_width):
                x2=expanded_x2
            else:
                x2=self.img_width


            expanded_y1 = y1 - add * height
            if(expanded_y1>0):
                y1=expanded_y1   
            else:
                y1=0

            expanded_y2 = y2 + add * height
            if(expanded_y2<self.img_height):
                y2=expanded_y2 
            else:
                y2=self.img_height


            expanded_boxes.append([x1, y1, x2, y2])
        expanded_boxes=np.array(expanded_boxes)

        return expanded_boxes




    def process_box_output(self, box_output):

        predictions = np.squeeze(box_output).T
        num_classes = box_output.shape[1] - self.num_masks - 4

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:4+num_classes], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], [], np.array([])

        box_predictions = predictions[..., :num_classes+4]
        mask_predictions = predictions[..., num_classes+4:]

        # Get the class with the highest confidence
        class_ids = np.argmax(box_predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(box_predictions)
        
        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)
        # print(boxes[indices])
        return boxes[indices], scores[indices], class_ids[indices], mask_predictions[indices]




    def extract_boxes(self, box_predictions):
        # Extract boxes from predictions
        boxes = box_predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes,
                                   (self.input_height, self.input_width),
                                   (self.img_height, self.img_width),divide=False)

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        # Check the boxes are within the image
        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)
        return boxes



    def crop_mask(self, masks, boxes):
        """
        It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box

        Args:
        masks (torch.Tensor): [h, w, n] tensor of masks
        boxes (torch.Tensor): [n, 4] tensor of bbox coordinates in relative point form

        Returns:
        (torch.Tensor): The masks are being cropped to the bounding box.
        """
        n, h, w = masks.shape
        x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
        r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
        c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)

        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))




    def process_mask_output(self, mask_predictions, mask_output):


        mask_output = np.squeeze(mask_output)

        mask_output = np.transpose(mask_output, (2, 0, 1))


        if mask_predictions.shape[0] == 0:
            return []

        mask_output = np.squeeze(mask_output)



        # Calculate the mask maps for each box
        num_mask, mask_height, mask_width = mask_output.shape  # CHW
        masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_height, mask_width))


        boxes_formask=self.boxes.copy()
        # print(boxes_formask)
        boxes_formask=self.expand_bounding_boxes(boxes_formask)
        # print(boxes_formask)

        #this line is making sure the bbox is add 15% to each size
        self.boxes=boxes_formask
        # Downscale the boxes to match the mask size
        scale_boxes = self.rescale_boxes(boxes_formask,
                                   (self.img_height, self.img_width),
                                   (mask_height, mask_width),divide=True)

        # For every box/mask pair, get the mask map

        mask_maps = np.zeros((len(scale_boxes), self.img_height, self.img_width))
        blur_size = (int(self.img_width / mask_width), int(self.img_height / mask_height))
        for i in range(len(scale_boxes)):

            scale_x1 = int(math.floor(scale_boxes[i][0]))
            scale_y1 = int(math.floor(scale_boxes[i][1]))
            scale_x2 = int(math.ceil(scale_boxes[i][2]))
            scale_y2 = int(math.ceil(scale_boxes[i][3]))

            x1 = int(math.floor(boxes_formask[i][0]))
            y1 = int(math.floor(boxes_formask[i][1]))
            x2 = int(math.ceil(boxes_formask[i][2]))
            y2 = int(math.ceil(boxes_formask[i][3]))

            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
            crop_mask = cv2.resize(scale_crop_mask,
                              (x2 - x1, y2 - y1),
                              interpolation=cv2.INTER_CUBIC)

            crop_mask = cv2.blur(crop_mask, blur_size)

            #use a loose bbox and low crop mask threshold, then feed more pixel to part segmentation model
            crop_mask = (crop_mask > 0.10).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask

        return mask_maps




    def draw_output(self, image, draw_scores=True, mask_alpha=0.5):

        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha, mask_maps=self.mask_maps, draw_bbox=False)


    def rescale_boxes(self, boxes, input_shape, image_shape, divide=True):
        # Rescale boxes to original image dimensions
        input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
        if divide:
            boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])
        return boxes