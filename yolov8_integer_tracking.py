
#environment for yolov8
import argparse
import time
import pdb
import os
import glob
import numpy as np
import cv2 as cv
import tflite_runtime.interpreter as tflite
#from tqdm import tqdm
#end of the environment for yolov8


#environment for tracking
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

# np.random.seed(0)

#end of tracking environment




#start of tracking code: 



def init_tracker():
    global tracker
    
    sort_max_age = 5 
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    tracker =Sort(max_age=sort_max_age,min_hits=sort_min_hits,iou_threshold=sort_iou_thresh)



def linear_assignment(cost_matrix):
    try:
        import lap #linear assignment problem solver
        _, x, y = lap.lapjv(cost_matrix, extend_cost = True)
        return np.array([[y[i],i] for i in x if i>=0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x,y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x,y)))


"""From SORT: Computes IOU between two boxes in the form [x1,y1,x2,y2]"""
def iou_batch(bb_test, bb_gt):
    
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[...,0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return(o)


"""Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form [x,y,s,r] where x,y is the center of the box and s is the scale/area and r is the aspect ratio"""
def convert_bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h    
    #scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


"""Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right"""
def convert_x_to_bbox(x, score=None):
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score==None):
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
    else:
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

"""This class represents the internal state of individual tracked objects observed as bbox."""
class KalmanBoxTracker(object):
    
    count = 0
    def __init__(self, bbox):
        """
        Initialize a tracker using initial bounding box
        
        Parameter 'bbox' must have 'detected class' int number at the -1 position.
        """
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10. # R: Covariance matrix of measurement noise (set to high for noisy inputs -> more 'inertia' of boxes')
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.5 # Q: Covariance matrix of process noise (set to high for erratically moving things)
        self.kf.Q[4:,4:] *= 0.5

        self.kf.x[:4] = convert_bbox_to_z(bbox) # STATE VECTOR
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.centroidarr = []
        CX = (bbox[0]+bbox[2])//2
        CY = (bbox[1]+bbox[3])//2
        self.centroidarr.append((CX,CY))
        
        #keep yolov5 detected class information
        self.detclass = bbox[5]

        # If we want to store bbox
        self.bbox_history = [bbox]
        
    def update(self, bbox):
        """
        Updates the state vector with observed bbox
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        self.detclass = bbox[5]
        CX = (bbox[0]+bbox[2])//2
        CY = (bbox[1]+bbox[3])//2
        self.centroidarr.append((CX,CY))
        self.bbox_history.append(bbox)
    
    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        # bbox=self.history[-1]
        # CX = (bbox[0]+bbox[2])/2
        # CY = (bbox[1]+bbox[3])/2
        # self.centroidarr.append((CX,CY))
    
        return self.history[-1]
    
    
    def get_state(self):
        """
        Returns the current bounding box estimate
        # test
        arr1 = np.array([[1,2,3,4]])
        arr2 = np.array([0])
        arr3 = np.expand_dims(arr2, 0)
        np.concatenate((arr1,arr3), axis=1)
        """
        arr_detclass = np.expand_dims(np.array([self.detclass]), 0)
        
        arr_u_dot = np.expand_dims(self.kf.x[4],0)
        arr_v_dot = np.expand_dims(self.kf.x[5],0)
        arr_s_dot = np.expand_dims(self.kf.x[6],0)
        
        return np.concatenate((convert_x_to_bbox(self.kf.x), arr_detclass, arr_u_dot, arr_v_dot, arr_s_dot), axis=1)
    
def associate_detections_to_trackers(detections, trackers, iou_threshold = 0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of 
    1. matches,
    2. unmatched_detections
    3. unmatched_trackers
    """

    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
    
    iou_matrix = iou_batch(detections, trackers)
    
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() ==1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))
    
    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)
    
    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]]<iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    
    if(len(matches)==0):
        matches = np.empty((0,2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
        
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
    

class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def getTrackers(self,):
        return self.trackers
        
    def update(self, dets= np.empty((0,6))):
        """
        Parameters:
        'dets' - a numpy array of detection in the format [[x1, y1, x2, y2, score], [x1,y1,x2,y2,score],...]
        
        Ensure to call this method even frame has no detections. (pass np.empty((0,5)))
        
        Returns a similar array, where the last column is object ID (replacing confidence score)
        
        NOTE: The number of objects returned may differ from the number of objects provided.
        """

        print(dets)
        self.frame_count += 1
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 6))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0, 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)
        
        # Update matched trackers with assigned detections
        for m in matched:

            self.trackers[m[1]].update(dets[m[0], :])
            
        # Create and initialize new trackers for unmatched detections
        for i in unmatched_dets:

            trk = KalmanBoxTracker(np.hstack((dets[i,:], np.array([0]))))
            #trk = KalmanBoxTracker(np.hstack(dets[i,:])
            self.trackers.append(trk)
        
        i = len(self.trackers)
        for trk in reversed(self.trackers):

            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1,-1)) #+1'd because MOT benchmark requires positive value
            i -= 1
            #remove dead tracklet
            if(trk.time_since_update >self.max_age):
                self.trackers.pop(i)
     

        #here i implemented a sorting algorithm for finding corresponding ones,
        #since the boundingbox found is constant, then this is also constant


        #so we have numpy array det
        #numpy array ret
        # print(ret)

        if(len(ret) > 0):

            return np.concatenate(ret)
        return np.empty((0,6))

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=1)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args

#end of tracking code






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
        # self.interpreter = tflite.Interpreter(model_path=f'models/{model_name}.tflite')
        self.interpreter = tflite.Interpreter(model_path='./yolov8l_integer_quant.tflite')
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
        self.conf_thres = 0.28
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

      

        # if object is not None:  
        #     results = [result for result in results if result['cls_name'] == object]

        # return results

        
        tracked_dets = tracker.update(results)
        # tracks =tracker.getTrackers()
        tracks =tracker.getTrackers()
        #cause tracked_dets is always [something, 9]
        # tracked_dets[:, 7]=
        # print(results.shape)
        # print(tracked_dets.shape)
        # track_result=np.concatenate((results,tracked_dets[:,8]),axis=1)
        temp=tracked_dets[:,8][::-1]
        # # print(temp.shape)
        temp=temp[:, None]
        # # print(temp.shape)

        track_result = np.concatenate((results, temp), axis=1)
        # print(track_result.shape)

        #so the track_result is a numpy array, each row is:
        #[a,b,c,d, confidence, class_cls, id]

        return track_result
    
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

                # for bbox, sscore in zip(box, score):
                #     result = {
                #         'bounding_box': bbox,
                #         'cls_id': i,
                #         'cls_name': coco_names[i],
                #         'score': sscore
                #     }
                #     results.append(result)
                # print(results)

            
                for bbox, sscore in zip(box, score):
                    result=np.hstack((bbox, sscore, i))
                    results.append(result)
        # print(results)
        
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

        #so the track_result(results) is a numpy array, each row is:
        #[a,b,c,d, confidence, cls_id, id]

        # print(results)


        for i in results:
            bbox=i[:4]
            confidence=i[4]
            cls_id=i[5]
            cls_name=coco_names[int(cls_id)]
            tracking_id=i[6]


        # for obj in results:
        #     bbox = obj['bounding_box']
        #     cls_id = obj['cls_id']
        #     cls_name = obj['cls_name']
        #     score = obj['score']

            label = f'{tracking_id}{" "+cls_name} {confidence:.2f}'
            color = self.colors(cls_id, True)

            im0 = self.plot_one_box(bbox, im0, color, label)

        cv.imwrite(save_path, im0)

#end of yolov8






if __name__ == '__main__':
    image_folder = '/home/myd/Desktop/test_frid'
    output_folder = './out/'
    

    init_tracker()

    yolo = YOLOV8()
    plotter = BboxesPlotter()

    image_files = glob.glob(f'{image_folder}/*.jpg')

    for file in image_files:
        start = time.time()
        
        

        results = yolo.detect(file)
        # print(tracker)
 
        print(f'Processing {file} - time: {time.time() - start} s')

        save_name = output_folder + file.split('/')[-1]
        plotter.plot_bboxes(file, results, save_name)
