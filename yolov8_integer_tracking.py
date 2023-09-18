
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
import random
# define a matrix
#A = array([[1, 2], [3, 4], [5, 6]])
#print(A)
# SVD
#U, s, VT = svd(A)
#print(U)
#print(s)
#print(VT)
#end of the environment for yolov8






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

        # print(output_data.shape)
        # flat_output=output_data.flatten()
        # random_integer = random.randint(1, 100)
        # print(random_integer)
        # np.savetxt(f'./random_tensor_{random_integer}.txt', flat_output, delimiter=' ')
        
        results = self.postprocess(output_data)


        #wont use tracking
        # tracked_dets = tracker.update(results)
        # tracks =tracker.getTrackers()
    
        # return tracked_dets

        #delete those small ones


        #2.97752563e+02 1.73334991e+02 3.35306946e+02 2.75268311e+02
        

        size_threshold=3872
        # print(results)
        final_result=[]
        test_image=cv.imread(image)

        for i in results:
            x=i[:4]
            x1,y1,x2,y2=map(int, x)
            detected=test_image[y1:y2, x1:x2]
            if (detected.shape[0]*detected.shape[1])>size_threshold:
                final_result.append(i)

        final_result=np.array(final_result)

        return final_result
    



    def postprocess(self, output_data):




        # #if we have a 84,8400
        # bbox2=[] #it should be 4*8400 
        # score2=[] #should be the rest
        # flat_output=output_data.flatten()

        # #traverse 80*8400
        # base=4*8400
        # #score2
        # m=0

        # while m<8400:
            
        #     temp2=[]
        #     i=0
        #     while i<4:
        #         temp2.append(flat_output[i*8400+m])
        #         i+=1
        #     temp2=np.array(temp2)
        #     bbox2.append(temp2)

        #     temp=[]
        #     n=0
        #     while n<80:
        #         temp.append(flat_output[n*8400+m+base])
        #         n+=1
        #     temp=np.array(temp)
        #     score2.append(temp)
        #     m+=1
        # bbox2=np.array(bbox2)
        # score2=np.array(score2)
        # print(bbox2.shape)
        # print(score2.shape)



        #correct ones
        output = np.squeeze(output_data).T
        boxes, probs = output[:, :4], output[:, 4:]


        # print(bbox2==boxes)
        # print(score2==probs)
        # print(bbox2.shape==boxes.shape)
        # print(score2.shape==probs.shape)


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

        # print(results)
       

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

            # print(detected.shape)

            #now using 9


            if detected.shape[0] and detected.shape[1]:




                split = 2  # Number of splits in each dimension (e.g., 3x3 grid)

                block_width = width // split
                block_height = height // split

                blocks = []

                for i in range(split):
                    for j in range(split):
                        block = detected[i * block_width: (i + 1) * block_width, j * block_height: (j + 1) * block_height]
                        # cv.imshow("block", block)
                        # cv.waitKey(0)
                        # cv.destroyAllWindows()
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

                # print(b)
                # print(g)
                # print(r)
                # print('-----------')



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
                # print(b_detected.shape,'-------------')
                # print(b_detected, g_detected, r_detected)

                # unique_id=np.hstack((10*(cls_id), b,g,r, confidence*100, x1/4,y1/4,x2/4,y2/4, b_detected/45, g_detected/45, r_detected/45))
                unique_id=np.hstack((10*(cls_id), b,g,r, confidence*100, x1/3,y1/3,x2/3,y2/3, b_detected/45, g_detected/45, r_detected/45))
                
                # print(unique_id)

                unique_ids.append(unique_id)



 

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


        print(ids1, ids2)


        return ids1, ids2




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

            label =f'{tracking_id} { confidence:.2f}'
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


        start2 = time.time()
        

        

        results2 = yolo.detect(file2)

        unique_ids2=yolo.output_id(file2,results2)
        

        save_name2 = output_folder + file2.split('/')[-1]
        
        # plotter.plot_bboxes(file, results, save_name)

        print(f'Processing {file2} - time: {time.time() - start2} s')


        start3 = time.time()

        # print(results1)
        # print('------')
        # print(results2)

        # print(unique_ids1)
        # print('-------------')
        # print(unique_ids2)
	

        ids1,ids2=yolo.compare(file1, results1, unique_ids1, file2, results2, unique_ids2)


        ## #get the threshold 3876 from image capture_1694334583895.png
        ## print(results1[0])
        ## m=results1[0][:4]
        ## x1, y1, x2, y2=map(int, m)
        ## image1=cv.imread(file1)
        ## detected=image1[y1:y2, x1:x2]
        ## threshold2=(y2-y1)*(x2-x1)
        ## print(threshold2)
        ## cv.imwrite("./1111.jpg",detected)



        plotter.plot_bboxes(file1, results1, save_name1, ids1)
        plotter.plot_bboxes(file2, results2, save_name2, ids2)

        print(f'Processing {file1, file2} - time: {time.time() - start3} s')


#opencv 4.5 tflite2.6

