
#environment for yolov8
import argparse
import time
import pdb
import os
import glob
import numpy as np
import cv2 as cv
import tflite_runtime.interpreter as tflite


# Singular-value decomposition
from numpy import array
from scipy.linalg import svd
import cv2
import numpy as np
import os
import uuid
# define a matrix
#A = array([[1, 2], [3, 4], [5, 6]])
#print(A)
# SVD
#U, s, VT = svd(A)
#print(U)
#print(s)
#print(VT)



#from tqdm import tqdm
#end of the environment for yolov8

# import random


# #environment for tracking
# import os
# import numpy as np
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from skimage import io

# import glob
# import time
# import argparse
# from filterpy.kalman import KalmanFilter



# # np.random.seed(0)

# #end of tracking environment




# #start of tracking code: 

# np.random.seed(0)

# def linear_assignment(cost_matrix):
#     try:
#         import lap #linear assignment problem solver
#         _, x, y = lap.lapjv(cost_matrix, extend_cost = True)
#         return np.array([[y[i],i] for i in x if i>=0])
#     except ImportError:
#         from scipy.optimize import linear_sum_assignment
#         x,y = linear_sum_assignment(cost_matrix)
#         return np.array(list(zip(x,y)))


# """From SORT: Computes IOU between two boxes in the form [x1,y1,x2,y2]"""
# def iou_batch(bb_test, bb_gt):
    
#     bb_gt = np.expand_dims(bb_gt, 0)
#     bb_test = np.expand_dims(bb_test, 1)
    
#     xx1 = np.maximum(bb_test[...,0], bb_gt[..., 0])
#     yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
#     xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
#     yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
#     w = np.maximum(0., xx2 - xx1)
#     h = np.maximum(0., yy2 - yy1)
#     wh = w * h
#     o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
#     + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
#     return(o)


# """Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form [x,y,s,r] where x,y is the center of the box and s is the scale/area and r is the aspect ratio"""
# def convert_bbox_to_z(bbox):
#     w = bbox[2] - bbox[0]
#     h = bbox[3] - bbox[1]
#     x = bbox[0] + w/2.
#     y = bbox[1] + h/2.
#     s = w * h    
#     #scale is just area
#     r = w / float(h)
#     return np.array([x, y, s, r]).reshape((4, 1))


# """Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
#     [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right"""
# def convert_x_to_bbox(x, score=None):
#     w = np.sqrt(x[2] * x[3])
#     h = x[2] / w
#     if(score==None):
#         return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
#     else:
#         return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

# """This class represents the internal state of individual tracked objects observed as bbox."""
# class KalmanBoxTracker(object):
    
#     count = 0
#     def __init__(self, bbox):
#         """
#         Initialize a tracker using initial bounding box
        
#         Parameter 'bbox' must have 'detected class' int number at the -1 position.
#         """
#         self.kf = KalmanFilter(dim_x=7, dim_z=4)
#         self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
#         self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

#         self.kf.R[2:,2:] *= 10. # R: Covariance matrix of measurement noise (set to high for noisy inputs -> more 'inertia' of boxes')
#         self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
#         self.kf.P *= 10.
#         self.kf.Q[-1,-1] *= 0.5 # Q: Covariance matrix of process noise (set to high for erratically moving things)
#         self.kf.Q[4:,4:] *= 0.5

#         self.kf.x[:4] = convert_bbox_to_z(bbox) # STATE VECTOR
#         self.time_since_update = 0
#         self.id = KalmanBoxTracker.count
#         KalmanBoxTracker.count += 1
#         self.history = []
#         self.hits = 0
#         self.hit_streak = 0
#         self.age = 0
#         self.centroidarr = []
#         CX = (bbox[0]+bbox[2])//2
#         CY = (bbox[1]+bbox[3])//2
#         self.centroidarr.append((CX,CY))
        
#         #keep yolov5 detected class information
#         self.detclass = bbox[5]

#         # If we want to store bbox
#         self.bbox_history = [bbox]
        
#     def update(self, bbox):
#         """
#         Updates the state vector with observed bbox
#         """
#         self.time_since_update = 0
#         self.history = []
#         self.hits += 1
#         self.hit_streak += 1
#         self.kf.update(convert_bbox_to_z(bbox))
#         self.detclass = bbox[5]
#         CX = (bbox[0]+bbox[2])//2
#         CY = (bbox[1]+bbox[3])//2
#         self.centroidarr.append((CX,CY))
#         self.bbox_history.append(bbox)
    
#     def predict(self):
#         """
#         Advances the state vector and returns the predicted bounding box estimate
#         """
#         if((self.kf.x[6]+self.kf.x[2])<=0):
#             self.kf.x[6] *= 0.0
#         self.kf.predict()
#         self.age += 1
#         if(self.time_since_update>0):
#             self.hit_streak = 0
#         self.time_since_update += 1
#         self.history.append(convert_x_to_bbox(self.kf.x))
#         # bbox=self.history[-1]
#         # CX = (bbox[0]+bbox[2])/2
#         # CY = (bbox[1]+bbox[3])/2
#         # self.centroidarr.append((CX,CY))
        
#         return self.history[-1]
    
    
#     def get_state(self):
#         """
#         Returns the current bounding box estimate
#         # test
#         arr1 = np.array([[1,2,3,4]])
#         arr2 = np.array([0])
#         arr3 = np.expand_dims(arr2, 0)
#         np.concatenate((arr1,arr3), axis=1)
#         """
#         arr_detclass = np.expand_dims(np.array([self.detclass]), 0)
        
#         arr_u_dot = np.expand_dims(self.kf.x[4],0)
#         arr_v_dot = np.expand_dims(self.kf.x[5],0)
#         arr_s_dot = np.expand_dims(self.kf.x[6],0)
        
#         return np.concatenate((convert_x_to_bbox(self.kf.x), arr_detclass, arr_u_dot, arr_v_dot, arr_s_dot), axis=1)
    
# def associate_detections_to_trackers(detections, trackers, iou_threshold = 0.3):
#     """
#     Assigns detections to tracked object (both represented as bounding boxes)
#     Returns 3 lists of 
#     1. matches,
#     2. unmatched_detections
#     3. unmatched_trackers
#     """
#     if(len(trackers)==0):
#         return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
    
#     iou_matrix = iou_batch(detections, trackers)
    
#     if min(iou_matrix.shape) > 0:
#         a = (iou_matrix > iou_threshold).astype(np.int32)
#         if a.sum(1).max() == 1 and a.sum(0).max() ==1:
#             matched_indices = np.stack(np.where(a), axis=1)
#         else:
#             matched_indices = linear_assignment(-iou_matrix)
#     else:
#         matched_indices = np.empty(shape=(0,2))
    
#     unmatched_detections = []
#     for d, det in enumerate(detections):
#         if(d not in matched_indices[:,0]):
#             unmatched_detections.append(d)
    
#     unmatched_trackers = []
#     for t, trk in enumerate(trackers):
#         if(t not in matched_indices[:,1]):
#             unmatched_trackers.append(t)
    
#     #filter out matched with low IOU
#     matches = []
#     for m in matched_indices:
#         if(iou_matrix[m[0], m[1]]<iou_threshold):
#             unmatched_detections.append(m[0])
#             unmatched_trackers.append(m[1])
#         else:
#             matches.append(m.reshape(1,2))
    
#     if(len(matches)==0):
#         matches = np.empty((0,2), dtype=int)
#     else:
#         matches = np.concatenate(matches, axis=0)
        
#     return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
    

# class Sort(object):
#     def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
#         """
#         Parameters for SORT
#         """
#         self.max_age = max_age
#         self.min_hits = min_hits
#         self.iou_threshold = iou_threshold
#         self.trackers = []
#         self.frame_count = 0
#     def getTrackers(self,):
#         return self.trackers
        
#     def update(self, dets= np.empty((0,6))):
#         """
#         Parameters:
#         'dets' - a numpy array of detection in the format [[x1, y1, x2, y2, score], [x1,y1,x2,y2,score],...]
        
#         Ensure to call this method even frame has no detections. (pass np.empty((0,5)))
        
#         Returns a similar array, where the last column is object ID (replacing confidence score)
        
#         NOTE: The number of objects returned may differ from the number of objects provided.
#         """
#         self.frame_count += 1
        
#         # Get predicted locations from existing trackers
#         trks = np.zeros((len(self.trackers), 6))
#         to_del = []
#         ret = []
#         for t, trk in enumerate(trks):
#             pos = self.trackers[t].predict()[0]
#             trk[:] = [pos[0], pos[1], pos[2], pos[3], 0, 0]
#             if np.any(np.isnan(pos)):
#                 to_del.append(t)
#         trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
#         for t in reversed(to_del):
#             self.trackers.pop(t)
#         matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)
        
#         # Update matched trackers with assigned detections
#         for m in matched:
#             self.trackers[m[1]].update(dets[m[0], :])
            
#         # Create and initialize new trackers for unmatched detections
#         for i in unmatched_dets:
#             trk = KalmanBoxTracker(np.hstack((dets[i,:], np.array([0]))))
#             #trk = KalmanBoxTracker(np.hstack(dets[i,:])
#             self.trackers.append(trk)
        
#         i = len(self.trackers)
#         for trk in reversed(self.trackers):
#             d = trk.get_state()[0]
#             if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
#                 ret.append(np.concatenate((d, [trk.id+1])).reshape(1,-1)) #+1'd because MOT benchmark requires positive value
#             i -= 1
#             #remove dead tracklet
#             if(trk.time_since_update >self.max_age):
#                 self.trackers.pop(i)


#         # print(dets)
#         # print(ret)


        
#         if(len(ret) > 0):
#             return np.concatenate(ret)
#         return np.empty((0,6))

# def parse_args():
#     """Parse input arguments."""
#     parser = argparse.ArgumentParser(description='SORT demo')
#     parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
#     parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
#     parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
#     parser.add_argument("--max_age", 
#                         help="Maximum number of frames to keep alive a track without associated detections.", 
#                         type=int, default=1)
#     parser.add_argument("--min_hits", 
#                         help="Minimum number of associated detections before track is initialised.", 
#                         type=int, default=3)
#     parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.03)
#     args = parser.parse_args()
#     return args






# tracker = None

# def init_tracker():
#     global tracker
    
#     sort_max_age = 5
#     sort_min_hits = 2
#     sort_iou_thresh = 0.1
#     tracker =Sort(max_age=sort_max_age,min_hits=sort_min_hits,iou_threshold=sort_iou_thresh)


# #end of tracking code






#start of yolov8

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
        # self.interpreter = tflite.Interpreter(model_path='./yolov8l_float32.tflite')
        # self.interpreter = tflite.Interpreter(model_path='./yolov8x6_float32.tflite')

        self.interpreter = tflite.Interpreter(model_path='./yolov8l_integer_quant.tflite')
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

        interpreter.invoke()

        output_data = interpreter.get_tensor(self.output_details[0]['index'])


        
        results = self.postprocess(output_data)

        #wont use tracking
        # tracked_dets = tracker.update(results)
        # tracks =tracker.getTrackers()
    
        # return tracked_dets

        #delete those small ones



        # print(results)




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

                    if (bbox[2]-bbox[0]!=0) and (bbox[3]-bbox[1]!=0):
                        result=np.hstack((bbox, sscore, i))
                        results.append(result)

        
        results=np.array(results)
       

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


    def normalize(self,image):
        # print(type(image))
        
        im = np.zeros((image.shape[0], image.shape[1], 3), dtype=float)
        for x in range(0,image.shape[0]):
            for y in range(0,image.shape[1]):
                pixel = image[x,y]
                #print("pixel was ", pixel)
                div = max(pixel[0],pixel[1], pixel[2])
                if div == 0:
                    div = 1
                lst = [float(pixel[0])/float(div), float(pixel[1])/float(div), float(pixel[2])/float(div)]
                im[x,y] = lst
                #print("image is ", im[x,y])

        return im

    #input a mxnx3 output a flat vector of the svd matrix
    def calculate_svd(self, detected):
        l1 = []
        l2 = []
        l3 = []
        l4 = []

        image1=yolo.normalize(detected)
        height, width, channels = image1.shape
        rc = image1[:, :, 0]  # Extract the first channel (red)
        gc = image1[:, :, 1]  # Extract the second channel (green)
        bc = image1[:, :, 2]  # Extract the third channel (blue)
    
        U, s, VT = svd(rc)
        U2, s2, VT2 = svd(gc)
        U3, s3, VT3 = svd(bc)
        # print(s[:5])
        # print(s2[:5])
        # print(s3[:5])
        l1.append(s[:5])
        l2.append(s2[:5])
        l3.append(s3[:5])
        l4.append((height,width))
        # print(l1)
        # print(l2)
        # print(l3)
        # print(l4)

        return  l1,l2,l3,l4


    def get_score(self, l1,l2,l3,l4,c1,c2,c3,c4):
        sum = 0
        mag2 = 0
        for x in range(0, len(l1)):
            sum += pow(pow(l1[x]-c1[x],2) + pow(l2[x]-c2[x],2) + pow(l3[x]-c3[x],2),0.5)

        for x in range(0, len(l4)):
            mag2 += pow(l4[x]-c4[x],2)

        mag2 = mag2 / pow((l4[0] * l4[1]) + (c4[0] * c4[1]),0.5)

        mag1 = pow(sum,0.5)
        mag2 = pow(mag2,0.5)
        # print(mag1, mag2)
        return mag1 + mag2


    def output_id(self, image, results):

        # load image
        if type(image) == str:  # Load from file path
            if not os.path.isfile(image):
                raise ValueError("Input image file path (", image, ") does not exist.")
            image = cv.imread(image)
        elif isinstance(image, np.ndarray):  # Use given NumPy array
            image = image.copy()
        else:
            raise ValueError("Invalid image input. Only file paths or a NumPy array accepted.")
        

        len_results=len(results)
        unique_ids=[]


        # output_directory = './detected/'
        # os.makedirs(output_directory, exist_ok=True)


    
        for i in range(len_results):

            cls_id=results[i][5]

            confidence=results[i][4]
            

            x=results[i][:4]


            x1, y1, x2, y2=map(int, x)

            width_25=int((y2-y1))
            height_25=int((x2-x1))


            if y1-width_25>0 and x1-height_25>0 and y2+width_25<image.shape[0] and x2+height_25<image.shape[1]:

                detected=image[(y1-width_25):(y2+width_25), (x1-height_25):(x2+height_25)]
            else:
                detected=image[y1:y2, x1:x2]


            # # cv.imwrite('./detected.jpg', detected) 

            # timestamp = int(time.time() * 1000)  # Use milliseconds for uniqueness
            # random_number = random.randint(1, 1000)  # Adjust the range as needed
            # random_filename = f'{timestamp}_{random_number}.jpg'  # Use the desired file extension
            # output_path = os.path.join(output_directory, random_filename)
            # cv.imwrite(output_path, detected)

            
            width=np.abs(y1-y2)
            height=np.abs(x1-x2)

                

            #now using 9


            if detected.shape[0] and detected.shape[1]:




                split = 2  # Number of splits in each dimension (e.g., 3x3 grid)

                block_width = width // split
                block_height = height // split

                blocks = []

                for i in range(split):
                    for j in range(split):
                        block = detected[i * block_width: (i + 1) * block_width, j * block_height: (j + 1) * block_height]
                        blocks.append(block)

                b=[]
                g=[]
                r=[]

                for i in range(split*split):
                    m1,m2,m3= cv.split(blocks[i])
                    b.append(m1)
                    g.append(m2)
                    r.append(m3)


                b_var=[]
                g_var=[]
                r_var=[]

                for i in range(split*split):
                    for j in range(i+1, split*split):
                        b_var.append(np.var(b[i])*np.var(b[j]))
                        g_var.append(np.var(g[i])*np.var(g[j]))
                        r_var.append(np.var(r[i])*np.var(r[j]))



                b=b_var/(width*height)
                b=b.astype(int)
                b=np.sort(b)

                g=g_var/(width*height)
                g=g.astype(int)
                g=np.sort(g)                
                
                r=r_var/(width*height)
                r=r.astype(int)
                r=np.sort(r)


            #     #rank

            #     # b_sort=np.sort(b)
            #     # # print(np.sort(b))
            #     # len_b_sort=len(b_sort)
    
            #     # for i in range(len_b_sort):
            #     #     for j in range(len(b)):
            #     #         # print(b[j])
            #     #         # print(b_sort[i])
            #     #         if b[j]==b_sort[i]:
            #     #             b[j]=(i+1)
                
            #     # g_sort=np.sort(g)
            #     # # print(np.sort(g))
            #     # len_g_sort=len(g_sort)
    
            #     # for i in range(len_g_sort):
            #     #     for j in range(len(g)):
            #     #         # print(g[j])
            #     #         # print(g_sort[i])
            #     #         if g[j]==g_sort[i]:
            #     #             g[j]=(i+1)

            #     # r_sort=np.sort(r)
            #     # # print(np.sort(r))
            #     # len_r_sort=len(r_sort)
    
            #     # for i in range(len_r_sort):
            #     #     for j in range(len(r)):
            #     #         # print(r[j])
            #     #         # print(r_sort[i])
            #     #         if r[j]==r_sort[i]:
            #     #             r[j]=(i+1)



                #bucketize

                b_max=np.max(b)

                b_min=np.min(b)
 
                binterval=(b_max-b_min)/(len(b)-1)


                for i in range(len(b)):
                    if binterval != 0:
                        b[i] = (b[i] - b_min) / binterval + 1
                    else:
                        # Handle the case where binterval is zero (or any other invalid value)
                        # You can assign a default value or raise an exception depending on your logic.
                        b[i] = 0 


                g_max=np.max(g)
                g_min=np.min(g)
                ginterval=(g_max-g_min)/(len(g)-1)
                # print(ginterval)
                for i in range(len(g)):
                    if ginterval !=0:
                        g[i]=(g[i]-g_min)/ginterval+1
                    else:
                        g[i]=0

                r_max=np.max(r)
                r_min=np.min(r)
                
                rinterval=(r_max-r_min)/(len(r)-1)
                # print(rinterval)
                for i in range(len(r)):
                    if rinterval !=0:
                        r[i]=(r[i]-r_min)/rinterval+1
                    else:
                        r[i]=0


                #add the color ratio

                b_detected, g_detected, r_detected= cv.split(detected)
                b_detected=np.var(b_detected)
                g_detected=np.var(g_detected)
                r_detected=np.var(r_detected)

                # unique_id=np.hstack((10*(cls_id), b,g,r, confidence*100, x1/4,y1/4,x2/4,y2/4, b_detected/45, g_detected/45, r_detected/45))
                unique_id=np.hstack((10*(cls_id), b,g,r, confidence*100, x1/3,y1/3,x2/3,y2/3, b_detected/45, g_detected/45, r_detected/45))
                
                unique_ids.append(unique_id)




                # #add svd, didnot work:
                # svd_vector=yolo.calculate_svd(detected)

                # svd_vector=svd_vector.reshape(svd_vector.shape[1])

                # unique_id=np.hstack((10*int(cls_id), b,g,r, 
                # confidence*100, x1/3,y1/3,x2/3,y2/3, 
                # b_detected/50, g_detected/50, r_detected/50,
                # 10*svd_vector))

                # unique_ids.append(unique_id)




                # split2=1000


                # b_mean=[]
                # g_mean=[]
                # r_mean=[]
                # for i in range(split):
                #     # block = detected[i * block_width: (i + 1) * block_width, 0: (j + 1) * height]
                #     block2=detected[0: width, i * block_height: (i + 1) * block_height]
                #     blocks.append(block)
                #     blocks.append(block2)

                # for i in blocks:
                #     bmean,gmean,rmean=cv.split(i)
                #     b_mean.append(bmean)
                #     g_mean.append(gmean)
                #     r_mean.append(rmean)

                # b_mean2=[]
                # g_mean2=[]
                # r_mean2=[]
                # for i in b_mean:
                #     b_mean2.append(np.var(i))
                # for i in g_mean:
                #     g_mean2.append(np.var(i))

                # for i in r_mean:
                #     r_mean2.append(np.var(i))

                # b_mean2=np.array(b_mean2)
                # g_mean2=np.array(g_mean2)
                # r_mean2=np.array(r_mean2)

                # b_mean2=np.sort(b_mean2)
                # g_mean2=np.sort(g_mean2)
                # r_mean2=np.sort(r_mean2)




                # b_mean2_max=np.max(b_mean2)

                # b_mean2_min=np.min(b_mean2)
 
                # binterval_mean=(b_mean2_max-b_mean2_min)/(len(b_mean2)-1)


                # for i in range(len(b_mean2)):
                #     if binterval_mean != 0:
                #         b_mean2[i] = (b_mean2[i] - b_mean2_min) / binterval_mean + 1
                #     else:
                #         # Handle the case where binterval is zero (or any other invalid value)
                #         # You can assign a default value or raise an exception depending on your logic.
                #         b_mean2[i] = 0 


                # g_mean2_max=np.max(g_mean2)

                # g_mean2_min=np.min(g_mean2)
 
                # ginterval_mean=(g_mean2_max-g_mean2_min)/(len(g_mean2)-1)


                # for i in range(len(g_mean2)):
                #     if ginterval_mean != 0:
                #         g_mean2[i] = (g_mean2[i] - g_mean2_min) / ginterval_mean + 1
                #     else:
                #         # Handle the case where ginterval is zero (or any other invalid value)
                #         # You can assign a default value or raise an exception depending on your logic.
                #         g_mean2[i] = 0 



                # r_mean2_max=np.max(r_mean2)

                # r_mean2_min=np.min(r_mean2)
 
                # rinterval_mean=(r_mean2_max-r_mean2_min)/(len(r_mean2)-1)


                # for i in range(len(r_mean2)):
                #     if rinterval_mean != 0:
                #         r_mean2[i] = (r_mean2[i] - r_mean2_min) / rinterval_mean + 1
                #     else:
                #         # Handle the case where binterval is zero (or any other invalid value)
                #         # You can assign a default value or raise an exception depending on your logic.
                #         r_mean2[i] = 0 


                # unique_id=np.hstack((10*int(cls_id), b,g,r,b_mean2,g_mean2,r_mean2, confidence*100, x1/3,y1/3,x2/3,y2/3))
                # unique_ids.append(unique_id)

 

        return unique_ids


    def compare(self, file1, results1, unique_ids1, file2, results2, unique_ids2):








        image1 = cv.imread(file1)
        image2 = cv.imread(file2)





        svd_threshold=8


        cut_threshold=40

        if len(unique_ids1)> len(unique_ids2):


            # ids1=np.arange(0, len(unique_ids1))
            ids1={i:[i,-1, unique_ids1[i][0]/10] for i in range(len(unique_ids1))}

            ids2 = {}

            addition=1

            # Iterate through the vectors in list2
            for i, vec2 in enumerate(unique_ids2):

                min_norm = float('inf')
                matching_id2 = -1

                # Compare with vectors in list1
                for j, vec1 in enumerate(unique_ids1):
                    if vec1[0]==vec2[0]:
                        norm = np.linalg.norm(vec1[1:]- vec2[1:])

                        if norm < min_norm:
                            min_norm = norm
                            matching_id2 = j
                    else:
                        nin_norm=-1
                        matching_id2-1

                if cut_threshold> min_norm:
                    ids2[i]=[matching_id2,min_norm, vec2[0]/10]
                    #the longer one should be [index, -1(for not selected by covariance, 1 for selected by covariance), class_id]
                    ids1[matching_id2][1]=1
                else:
                    ids2[i]=[-1,-1,  vec2[0]/10]
                    



            for key1,value1 in ids2.items():

                for key2,value2 in ids2.items():
                    if key1!=key2 and value1[0]==value2[0] and value1[0]!=-1:
                        if value1[2]==value2[2]:
                            if value1[1]!=-1 and value2[1]!=-1:
                                if value1[1]>value2[1]:
                                        value2[0]=len(unique_ids1)+addition
                                        addition+=1
                                else:
                                    value1[0]=len(unique_ids1)+addition
                                    addition+=1
                        else:
                            if value1[1]!=-1 and value2[1]!=-1:
                                #so two with different class is now the same number, add either one with more index
                                value1[0]=len(unique_ids1)+addition+1
                                addition+=1





            

            for i in range(len(ids2)):
                if ids2[i][0]==-1:
                    x2=results2[i][:4]
                    x21, y21, x22, y22=map(int, x2)
                    detected2=image2[y21:y22, x21:x22]
                    
                    class_id2=results2[i][5]
                    # print(results1[i][5], ids1[i][2])

                
                    index=-1
                    min_score=1000000
                    for j in range(len(ids1)):
                        if ids1[j][1]==-1:
                            x=results1[j][:4]
                            x1, y1, x2, y2=map(int, x)
                            detected1=image1[y1:y2, x1:x2]
                            class_id1=results1[j][5]
                            l1,l2,l3,l4=yolo.calculate_svd(detected2)

                            

                            if class_id1==class_id2:
                                
                                c1,c2,c3,c4=yolo.calculate_svd(detected1)
                                
                                for x in range(0,len(l1)):
                                    
                                    for y in range(0,len(c1)):
                                        ms = yolo.get_score(l1[x],l2[x],l3[x],l4[x],c1[y],c2[y],c3[y],c4[y])
                                        if ms < min_score:
                                            min_score = ms
                                            index=j
                                            # print(i,min_score, index)


                                        if min_score<svd_threshold:
                                            ids2[i][0]=index
                                            ids2[i][1]=-2
            # print('11111111111')

            # print(ids1)
            # print(ids2)



        else:
            ids2={i:[i,-1, unique_ids2[i][0]/10] for i in range(len(unique_ids2))}

            ids1 ={}

            addition=1


            # Iterate through the vectors in list1
            

            for i, vec1 in enumerate(unique_ids1):
                min_norm1 = float('inf')
                matching_id = -1

                # Compare with vectors in list2
                
                for j, vec2 in enumerate(unique_ids2):

                    
                    if vec1[0]==vec2[0]:

                        norm = np.linalg.norm(vec2[1:]- vec1[1:])


                        if norm < min_norm1:
                            min_norm1 = norm
                            matching_id = j


                if cut_threshold> min_norm1:
                    ids1[i]=[matching_id,min_norm1,  vec1[0]/10]
                    ids2[matching_id][1]=1
                else:
                    ids1[i]=[-1,-1, vec1[0]/10]

            # print(ids1, ids2,'---------------------')

            for key1,value1 in ids1.items():

                for key2,value2 in ids1.items():
                    if key1!=key2 and value1[0]==value2[0]:
                            # print(ids1[key1][0],ids1[key2][0])
                        if value1[2]==value2[2]:
                            if value1[1]!=-1 and value2[1]!=-1:
                                if value1[1]>value2[1]:
                                    value2[0]=len(unique_ids2)+addition
                                    addition+=1
                                else:
                                    value1[0]=len(unique_ids2)+addition
                                    addition+=1
                        else:
                            if value1[1]!=-1 and value2[1]!=-1:
                                value1[0]=len(unique_ids2)+addition+1
                                addition+=1
        #here remeber ids2 is longer



            for i in range(len(ids1)):
                if ids1[i][0]==-1:
                    x=results1[i][:4]
                    x1, y1, x2, y2=map(int, x)
                    detected1=image1[y1:y2, x1:x2]
                    
                    class_id1=results1[i][5]
                    # print(results1[i][5], ids1[i][2])

                
                    index=-1
                    min_score=1000000
                    for j in range(len(ids2)):
                        if ids2[j][1]==-1:
                            x2=results2[j][:4]
                            x21, y21, x22, y22=map(int, x2)
                            detected2=image2[y21:y22, x21:x22]
                            class_id2=results2[j][5]
                            l1,l2,l3,l4=yolo.calculate_svd(detected1)

                            

                            if class_id1==class_id2:
                                
                                c1,c2,c3,c4=yolo.calculate_svd(detected2)
                                
                                for x in range(0,len(l1)):
                                    
                                    for y in range(0,len(c1)):
                                        ms = yolo.get_score(l1[x],l2[x],l3[x],l4[x],c1[y],c2[y],c3[y],c4[y])
                                        if ms < min_score:
                                            min_score = ms
                                            index=j
                                            # print(min_score, index)


                                        if min_score<svd_threshold:
                                            ids1[i][0]=index
                                            ids1[i][1]=-2


 


        return ids1, ids2




    # ssim not working because the definition of covariance is positional encoding 
    # def ssim(self, image1, results1,bit_depth1, image2, results2, bit_depth2):

    #     detected1=[]
    #     detected2=[]

    #     for i in range(len(results1)):
    #         x=results1[i][:4]
    #         x1,y1,x2,y2=map(int, x)
    #         detected1.append(image1[y1:y2, x1:x2])
    #         # mean, stddev = cv.meanStdDev(image1[y1:y2, x1:x2])
    #         # print(mean, stddev)
    #         # cv.imshow('111',image1[y1:y2, x1:x2])
    #         # cv.waitKey(0)

    #     print('second')

    #     for i in range(len(results2)):
    #         y=results2[i][:4]
    #         x_1,y_1,x_2,y_2=map(int, y)
    #         detected2.append(image2[y_1:y_2, x_1:x_2])
    #         # mean, stddev = cv.meanStdDev(image2[y1:y2, x1:x2])
    #         # print(mean, stddev)
    #         # cv.imshow('111',image2[y_1:y_2, x_1:x_2])
    #         # cv.waitKey(0)



    #     if len(detected1)>len(detected2):
            
    #         print('111111111111')

    #         ids1={i:[i,-1] for i in range(len(detected1))}

    #         ids2={}

    #         for i, detection_i in enumerate(detected2):
    #             max_ssim=0
    #             ssim=0
    #             matching_id2=-1
    #             for j, detection_j in enumerate(detected1):
    #                 if results1[j][5]==results2[i][5]:
    #                     ssim=self.ssim_helper1(detection_i, bit_depth2, detection_j, bit_depth1)
    #                     if ssim > max_ssim:
                            
    #                         max_ssim=ssim
    #                         matching_id2=j
    #                         # print(j)
    #                         # print('aaa',matching_id2, max_ssim)
                
    #             # print('11111')

    #             if i in ids2:
    #                 if ids2[i][1]<max_ssim:
    #                     ids2[i]=[matching_id2, max_ssim]
    #             else:
    #                 ids2[i]=[matching_id2,max_ssim]



    #     else:


    #         ids2={i:[i,-1] for i in range(len(detected2))}

    #         ids1={}

    #         for i, detection_i in enumerate(detected1):
    #             max_ssim=0
    #             matching_id=-1

    #             for j, detection_j in enumerate(detected2):

    #                 if results1[i][5]==results2[j][5]:

    #                     ssim=self.ssim_helper1(detection_i, bit_depth1, detection_j, bit_depth2)

    #                     if ssim > max_ssim:
    #                         max_ssim=ssim

    #                         matching_id=j


    #             if i in ids1:

    #                 if ids1[i][1]<max_ssim:

    #                     ids1[i]=[matching_id, max_ssim]
    #             else:
    #                 ids1[i]=[matching_id,max_ssim]

    #     return ids1, ids2


        
    # def ssim_helper1(self, detected1, bit_depth1, detected2, bit_depth2):
    #     b1, g1, r1 = cv.split(detected1)
    #     b2, g2, r2 = cv.split(detected2)

    #     b=self.ssim_helper2(b1, bit_depth1, b2, bit_depth2)
    #     g=self.ssim_helper2(g1, bit_depth1, g2, bit_depth2)
    #     r=self.ssim_helper2(r1, bit_depth1, r2, bit_depth2)

    #     return np.linalg.norm(np.array([b,g,r]))
        
        

    # def ssim_helper2(self, x,bit_depth1, y, bit_depth2):


    #     stddev_x=np.std(x)
    #     stddev_y=np.std(y)

    #     mean_x = np.mean(x, axis=(0, 1))
    #     mean_y = np.mean(y, axis=(0, 1))

    #     # print(stddev_x, stddev_y)
    #     # print(mean_x, mean_y)

    #     variance_x = np.var(x, axis=(0, 1))
    #     variance_y = np.var(y, axis=(0, 1))

    #     # covariance_matrix = np.cov(np.vstack((x.reshape(-1, 3).T, y.reshape(-1, 3).T)))

    #     # covariance_xy = covariance_matrix[:3, 3:]
    #     covariance_xy=(variance_x * variance_y) - (mean_x * mean_y)


    #     # print("Pixel Sample Mean of x:", mean_x)
    #     # print("Pixel Sample Mean of y:", mean_y)
    #     # print("Variance of x:", variance_x)
    #     # print("Variance of y:", variance_y)
    #     # print("Covariance between x and y:")
    #     # print(covariance_xy)

    #     L1=2*bit_depth1 -1
    #     L2=2*bit_depth2-1

    #     k1=0.01
    #     k2=0.03

    #     c1=(k1*L1)*(k1*L1)
    #     c2=(k2*L2)*(k2*L2)

    #     c3=c2/2

    #     l_xy=(2*mean_x*mean_y+c1)/(np.square(mean_x)+np.square(mean_y)+c1)

    #     c_xy=(2*variance_x*variance_y+c1)/(np.square(mean_x)+np.square(mean_y)+c2)

    #     s_xy=(covariance_xy+c3)/(stddev_x*stddev_y+c3)


    #     return l_xy*c_xy*s_xy






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

    def plot_bboxes(self, img_path, results, save_path=None, id=None):
        
        im0 = cv.imread(img_path)
        # im0=img_path

        for i,value in enumerate(results):
            bbox=value[:4]
            confidence=value[4]
            # track_id=i[8]
            cls_id=value[5]
            cls_name=coco_names[int(cls_id)]

            tracking_id=id[i][0]

            # label = f'{tracking_id} {confidence:.2f}'
            # label = f'{tracking_id} {confidence:.2f}'
            label =f'{tracking_id}'
            color = self.colors(cls_id, True)

            im0 = self.plot_one_box(bbox, im0, color, label)

        try:
            cv.imwrite(save_path, im0)
        except Exception as e:
            print("Error:", e)

        # cv.imshow("image", im0)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        #if it is using a vide, comment the image output out then return the im0
        # return im0



#end of yolov8






##########################single image tracking using covariance


if __name__ == '__main__':


    for x in range(1,2):
        image_folder = './test/test_case'+str(x)
        output_folder = './out/'
        

        # init_tracker()

        yolo = YOLOV8()
        plotter = BboxesPlotter()

        image_files = glob.glob(f'{image_folder}/*.[jp][pn][ge]')
        sorted_image_files = sorted(image_files)

        
    
        file1=sorted_image_files[0]
        #should iterative twice

        start1 = time.time()
        
        

        results1 = yolo.detect(file1)

        unique_ids1=yolo.output_id(file1,results1)

        save_name1 = output_folder + file1.split('/')[-1]

        # plotter.plot_bboxes(file, results, save_name)

        print(f'Processing {file1} - time: {time.time() - start1} s')


        file2=sorted_image_files[1]
        #should iterative twice

        start2 = time.time()
        
        

        results2 = yolo.detect(file2)

        unique_ids2=yolo.output_id(file2,results2)
        

        save_name2 = output_folder + file2.split('/')[-1]
        
        # plotter.plot_bboxes(file, results, save_name)

        print(f'Processing {file2} - time: {time.time() - start2} s')


        start3 = time.time()

        # print(len(unique_ids1))
        # print('--------------')
        # print(len(unique_ids2))

        ids1,ids2=yolo.compare(file1, results1, unique_ids1, file2, results2, unique_ids2)

        # print(results1.shape)
        # print(results2.shape)

        # print(ids1)
        # print(ids2)

        plotter.plot_bboxes(file1, results1, save_name1, ids1)
        plotter.plot_bboxes(file2, results2, save_name2, ids2)

        print(f'Processing {file1, file2} - time: {time.time() - start3} s')


#opencv 4.5 tflite2.6


###########################################video no tracking
# if __name__ == '__main__':



#     if tf.test.is_gpu_available():
#         print("GPU is available.")
#     else:
#         print("GPU is not available.")


#     video_path = '/home/myd/Desktop/30min.mp4'  # Change this to your video file path
#     output_path = '/home/myd/Desktop/out/video.mp4'  # Change this to your desired output video path
#     fps = 1.0  # Process one frame per second

#     yolo = YOLOV8()
#     plotter = BboxesPlotter()

#     cap = cv.VideoCapture(video_path)

#     if not cap.isOpened():
#         print("Error: Could not open video file.")
#         exit()

#     frame_rate = cap.get(cv.CAP_PROP_FPS)
#     h, w = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
#     fourcc = cv.VideoWriter_fourcc(*'XVID')
#     video_writer = cv.VideoWriter(output_path, fourcc, frame_rate, (w, h))

#     while True:
#         ret, frame = cap.read()

#         if not ret:
#             break

#         results = yolo.detect(frame)

#         # Annotate the frame
  
#         annotated_frame=plotter.plot_bboxes(frame, results)
#         #cv.imwrite('/home/myd/Desktop/out/eg.jpg',annotated_frame)

#         # Save the annotated frame to the output video
 
#         video_writer.write(annotated_frame)

#     cap.release()
#     video_writer.release()

#     print("Processing completed. Annotated video saved at:", output_path)
    
    








 

# ###########################################just image inference

# # if __name__ == '__main__':
# #     for scene in ['bus', 'school', 'hospital', 'dwight']:
# #         files = glob.glob(f'src/{scene}/*.jpg')[:5]

# #         yolo = YOLOV8()
# #         plotter = BboxesPlotter()

# #         for file in files:
            
# #             start = time.time()
# #             results = yolo.detect(file)
# #             print(f'time: {time.time() - start} s')

# #             save_name = f'out/{model_name}/' + f'{scene}_' + file.split('/')[-1]
# #             dir_name = os.path.dirname(save_name)
# #             if not os.path.exists(dir_name):
# #                 os.makedirs(dir_name)
            
# #             plotter.plot_bboxes(file, results, save_name)

# if __name__ == '__main__':
#     image_folder = '/home/myd/Desktop/multi'
#     output_folder = './out/'

#     yolo = YOLOV8()
#     plotter = BboxesPlotter()

#     image_files = glob.glob(f'{image_folder}/*.png')

#     for file in image_files:
#         start = time.time()
#         results = yolo.detect(file)

#         print(f'Processing {file} - time: {time.time() - start} s')

#         save_name = output_folder + file.split('/')[-1]











###################ssim one, not working right now

# if __name__ == '__main__':
#     image_folder = '/home/myd/Desktop/baseball'
#     output_folder = './out/'
    

#     # init_tracker()

#     yolo = YOLOV8()
#     plotter = BboxesPlotter()

#     image_files = glob.glob(f'{image_folder}/*.jpg')
#     sorted_image_files = sorted(image_files)

    
    
#     file1=sorted_image_files[0]
#     #should iterative twice

#     start1 = time.time()
    
    

#     results1 = yolo.detect(file1)

    

#     save_name1 = output_folder + file1.split('/')[-1]
#     # plotter.plot_bboxes(file, results, save_name)

#     print(f'Processing {file1} - time: {time.time() - start1} s')





#     file2=sorted_image_files[1]
#     #should iterative twice

#     start2 = time.time()
    
    

#     results2 = yolo.detect(file2)


    

#     # print(results2)
#     # print(unique_ids2)

#     save_name2 = output_folder + file2.split('/')[-1]
#     # plotter.plot_bboxes(file, results, save_name)

#     print(f'Processing {file2} - time: {time.time() - start2} s')

    
#     #now i have image1, iamge2, resutls1, results2
    




#     # start3 = time.time()


#     # if len(unique_ids1)> len(unique_ids2):

#     #     # ids1=np.arange(0, len(unique_ids1))
#     #     ids1={i:[i,-1] for i in range(len(unique_ids1))}

#     #     ids2 = {}

#     #     # Iterate through the vectors in list2
#     #     for i, vec2 in enumerate(unique_ids2):
#     #         min_norm = float('inf')
#     #         matching_id2 = -1

#     #         # Compare with vectors in list1
#     #         for j, vec1 in enumerate(unique_ids1):
#     #             if vec1[0]==vec2[0]:
#     #                 norm = np.linalg.norm(vec1[1:]- vec2[1:])
#     #                 if norm < min_norm:
#     #                     min_norm = norm
#     #                     matching_id2 = j

#     #         # Assign the same unique ID for the closest vector in list1
#     #         if i in ids2:
#     #             if ids2[i][1]>min_norm:
#     #                 ids2[i]=[matching_id2,min_norm]
                
#     #         else:
#     #             ids2[i]=[matching_id2,min_norm]

            
#     # else:
#     #     ids2={i:[i,-1] for i in range(len(unique_ids2))}

#     #     ids1 ={}


#     #     # Iterate through the vectors in list1
#     #     for i, vec1 in enumerate(unique_ids1):
#     #         min_norm = float('inf')
#     #         matching_id = -1

#     #         # Compare with vectors in list2
#     #         # print(vec1.shape)
#     #         for j, vec2 in enumerate(unique_ids2):

#     #             if vec1[0]==vec2[0]:
#     #                 norm = np.linalg.norm(vec1[1:]- vec2[1:])
#     #                 if norm < min_norm:
#     #                     min_norm = norm
#     #                     matching_id = j

#     #         # Assign the same unique ID for the closest vector in list2
#     #         if i in ids1:
#     #             if ids1[i][1]>min_norm:
#     #                 ids1[i]=[matching_id,min_norm]
#     #         else:
                
#     #             ids1[i]=[matching_id,min_norm]
        
#     # # print(ids1)
#     # # print(ids2)





#     im1=cv.imread(file1)
#     im2=cv.imread(file2)
#     # print(im1.shape)
#     # print(im2.shape)
#     # print(results1)
#     # print(results2)

#     bit_depth1=im1.dtype.itemsize * 8
#     bit_depth2=im2.dtype.itemsize * 8

#     assert bit_depth1 == bit_depth2, "Error: Bit depths of the two images are not equal."

#     # print(f'Processing {file1, file2} - time: {time.time() - start3} s')
#     ids1, ids2= yolo.ssim(im1, results1,bit_depth1, im2, results2, bit_depth2)

#     print(ids1)

#     print(ids2)


#     plotter.plot_bboxes(file1, results1, save_name1, ids1)
#     plotter.plot_bboxes(file2, results2, save_name2, ids2)
