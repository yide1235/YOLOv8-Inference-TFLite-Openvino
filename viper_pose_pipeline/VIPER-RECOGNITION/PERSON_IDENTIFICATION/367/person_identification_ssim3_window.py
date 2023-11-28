# Import required packages/scripts
import numpy as np
import cv2
import sys
import os
sys.path.append('./LINE_MATCHING/')
sys.path.append('./YOLOV8')
# sys.path.append('../../REALTIME7/VIPER-REGISTRAR/lib/camera_calibration')
sys.path.append('./THUNDER/')
sys.path.append('./TRIANGULATION')
import csv
from yolov8 import run_yolo
from thunder import run_thunder
from line_matching_4 import sweep_line_block
from triangulate_point import triangulate_points
import random


import numpy as np
import math



def adjust_bbox(bbox):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    # Find the larger side and round it up to the nearest multiple of 50
    new_size = math.ceil(max(width, height) / 40) * 40

    # Calculate new coordinates to keep the bbox centered
    center_x, center_y = x1 + width / 2, y1 + height / 2
    new_x1 = center_x - new_size / 2
    new_y1 = center_y - new_size / 2
    new_x2 = new_x1 + new_size
    new_y2 = new_y1 + new_size

    return np.array([new_x1, new_y1, new_x2, new_y2])




def person_vecs_identification(count, mtxl,mtxr,R,T,H1,H2,imgl,imgr,verbose=False, shift=30):


    imgl_rectified = cv2.warpPerspective(imgl, H1, (1920,1080))
    imgr_rectified = cv2.warpPerspective(imgr, H2, (1920,1080))

    #only use for ssim
    im_l_float32=imgl_rectified.astype('float32')/255.0
    im_r_float32=imgr_rectified.astype('float32')/255.0
    #only use for ssim

    cv2.imwrite('tmp_r.jpg',imgr_rectified)
    cv2.imwrite('tmp_l.jpg',imgl_rectified)
    # cv2.imwrite('tmp_save.jpg',imgr_rectified) #here looks good

    bboxes_r=run_yolo(['tmp_r.jpg'])
    bboxes_l=run_yolo(['tmp_l.jpg'])

    # os.remove('tmp_r.jpg')
    # os.remove('tmp_l.jpg')

    # print(bboxes_r,bboxes_l)
    

    variance_threshold=200


    #doing random data augmentation for the right rectified images
    rectified_right_points_xlist=None
    rectified_right_points_ylist=None


    imgr_rectified = cv2.imread('tmp_r.jpg')




    for g in range(50):
      # Step 1: Load the image

      height, width = imgr_rectified.shape[:2]

      imgr_temp = imgr_rectified.copy()
      # Step 2: Apply random augmentation
      # Randomly choose an augmentation: noise, stretching, or cropping
      augmentation = random.choice(['noise', 'stretch','crop'])
      # augmentation = random.choice(['noise', 'stretch'])

      # Initialize transformation matrix
      M = np.eye(3)

      if augmentation == 'noise':
          # Check the number of channels in the image
          if len(imgr_temp.shape) == 3:
              # If the image is color (BGR), create noise with 3 channels
              noise = np.random.randint(0, 50, imgr_temp.shape, dtype='uint8')
          else:
              # If the image is grayscale, create single channel noise
              noise = np.random.randint(0, 50, (height, width), dtype='uint8')

          imgr_temp = cv2.add(imgr_temp, noise)

      elif augmentation == 'stretch':
          # Randomly choose stretch factors
          fx, fy = random.uniform(0.8, 1.2), random.uniform(0.8, 1.2)
          imgr_temp = cv2.resize(imgr_temp, None, fx=fx, fy=fy)
          M = np.array([[fx, 0, 0], [0, fy, 0], [0, 0, 1]])

      elif augmentation == 'crop':
          # Randomly choose crop size
          x, y = random.randint(0, width // 4), random.randint(0, height // 4)
          w, h = width - 2*x, height - 2*y
          imgr_temp = imgr_temp[y:y+h, x:x+w]
          M = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]])

      # Step 3: Detect keypoints using the augmented image
      rectified_right_points = []

      for bbox in bboxes_r:
          # Extract and save the cropped image from the bounding box
          cropped_img = imgr_temp[max(0, int(bbox[1] - 0.05 * (bbox[3] - bbox[1]))):int(min(imgr_temp.shape[0], bbox[3] + 0.05 * (bbox[3] - bbox[1]))), max(0, int(bbox[0] - 0.05 * (bbox[2] - bbox[0]))):min(imgr_temp.shape[1], int(bbox[2] + 0.05 * (bbox[2] - bbox[0])))]
          cv2.imwrite('box_r.jpg', cropped_img)

          p_r = run_thunder('box_r.jpg')

          for point in p_r[0]:
              if point[0] == point[1] == 0:
                  points_r = [[0, 0]]
              else:
                  # Adjust keypoints based on augmentation
                  x_adj, y_adj = point[0] + max(0, int(bbox[0] - 0.05 * (bbox[2] - bbox[0]))), point[1] + max(0, int(bbox[1] - 0.05 * (bbox[3] - bbox[1])))

                  # Transform points back according to the inverse of M
                  if augmentation != 'noise':
                      inv_M = np.linalg.inv(M)
                      x_adj, y_adj, _ = np.dot(inv_M, [x_adj, y_adj, 1])

                  points_r = [[int(x_adj), int(y_adj)]]

              rectified_right_points.append(points_r)




      temp_right=np.array(rectified_right_points).reshape(17,2)



      temp_right_x=temp_right[:,0]
      temp_right_y=temp_right[:,1]



      if rectified_right_points_xlist is None:
        rectified_right_points_xlist=temp_right_x
      else:
        rectified_right_points_xlist=np.vstack((rectified_right_points_xlist, temp_right_x))



      if rectified_right_points_ylist is None:
        rectified_right_points_ylist=temp_right_y
      else:
        rectified_right_points_ylist=np.vstack((rectified_right_points_ylist, temp_right_y))


    #end of 50:


    rectified_right_points_xlist = rectified_right_points_xlist.astype(float).T
    rectified_right_points_ylist = rectified_right_points_ylist.astype(float).T


    #now you get 17,100

    rectified_right_points_xlist=rectified_right_points_xlist.tolist()
    rectified_right_points_ylist=rectified_right_points_ylist.tolist()

    rectified_right_points_xlist_clean = [[value for value in sublist if value != 0.0] for sublist in rectified_right_points_xlist]
    rectified_right_points_ylist_clean = [[value for value in sublist if value != 0.0] for sublist in rectified_right_points_ylist]


    for i in range(len(rectified_right_points_xlist_clean)):
      rectified_right_points_xlist_clean[i]= sorted(rectified_right_points_xlist_clean[i])
      n=len(rectified_right_points_xlist_clean[i])
      lower_index = int(np.ceil(n * 0.25))
      upper_index = int(np.floor(n * 0.75))
      rectified_right_points_xlist_clean[i]=rectified_right_points_xlist_clean[i][lower_index:upper_index]

    

    for i in range(len(rectified_right_points_ylist_clean)):
      rectified_right_points_ylist_clean[i]= sorted(rectified_right_points_ylist_clean[i])
      n=len(rectified_right_points_ylist_clean[i])
      lower_index = int(np.ceil(n * 0.25))
      upper_index = int(np.floor(n * 0.75))
      rectified_right_points_ylist_clean[i]=rectified_right_points_ylist_clean[i][lower_index:upper_index]



    #delete the high variance ones
    for i in range(len(rectified_right_points_xlist_clean)):
      # print(np.var(rectified_right_points_xlist_clean[i]))
      if np.var(rectified_right_points_xlist_clean[i])>variance_threshold:
        rectified_right_points_xlist_clean[i]=[0]

    for i in range(len(rectified_right_points_ylist_clean)):
      # print(np.var(rectified_right_points_ylist_clean[i]))
      if np.var(rectified_right_points_ylist_clean[i])>variance_threshold:
        rectified_right_points_ylist_clean[i]=[0]



    

    for i in range(len(rectified_right_points_xlist_clean)):
      # #this is the mean
      # i_mean=sum(rectified_right_points_xlist_clean[i])/len(rectified_right_points_xlist_clean[i])
      # rectified_right_points_xlist_clean[i]=i_mean
      
      
      #get the median:

      len_n=len(rectified_right_points_xlist_clean[i])
      # print(len_n)
      if len_n>=2:
        if len_n%2 ==1:
          rectified_right_points_xlist_clean[i]=rectified_right_points_xlist_clean[i][len_n//2]
        else:
          mid1=rectified_right_points_xlist_clean[i][len_n // 2 - 1 ]
          mid2=rectified_right_points_xlist_clean[i][len_n // 2]
          rectified_right_points_xlist_clean[i]=(mid1+mid2)/2

      elif len_n==1:
        rectified_right_points_xlist_clean[i]=rectified_right_points_xlist_clean[i][0]
      else:
        #len_n=0
        rectified_right_points_xlist_clean[i]=0




    for i in range(len(rectified_right_points_ylist_clean)):
      # #mean
      # i_mean=sum(rectified_right_points_ylist_clean[i])/len(rectified_right_points_ylist_clean[i])
      # rectified_right_points_ylist_clean[i]=i_mean

      #median:
      len_n=len(rectified_right_points_ylist_clean[i])
      if len_n>=2:

        if len_n%2 ==1:
          rectified_right_points_ylist_clean[i]=rectified_right_points_ylist_clean[i][len_n//2]
        else:
          mid1=rectified_right_points_ylist_clean[i][len_n // 2 - 1 ]
          mid2=rectified_right_points_ylist_clean[i][len_n // 2]
          rectified_right_points_ylist_clean[i]=(mid1+mid2)/2
      elif len_n==1:
         rectified_right_points_ylist_clean[i]=rectified_right_points_ylist_clean[i][0]
      else:
        #len_n=0
        rectified_right_points_ylist_clean[i]=0



    result_right=[]


    assert len(rectified_right_points_xlist_clean)==len(rectified_right_points_ylist_clean)
    for i in range(len(rectified_right_points_xlist_clean)):
      result_right.append([[rectified_right_points_xlist_clean[i],rectified_right_points_ylist_clean[i]]])


    rectified_right_points_result=result_right
    #end of right




    #-----------------------------------------------------


    # normal finding left points
    # Loop through points, align corresponding left and  lines from rectified images, and get left points
    rectified_left_points_result=[]
    # print('box: ', bboxes_l)
    for bbox in bboxes_l:

        cv2.imwrite('box_l.jpg',imgl_rectified[max(0,int(bbox[1]-.05*(bbox[3]-bbox[1]))):int(min(imgl_rectified.shape[0],bbox[3]+.05*(bbox[3]-bbox[1]))),max(0,int(bbox[0]-.05*(bbox[2]-bbox[0]))):min(imgl_rectified.shape[1],int(bbox[2]+.05*(bbox[2]-bbox[0])))])

        p_l=run_thunder('box_l.jpg')

        for point in p_l[0]:
          if point[0]==point[1]==0:
            points_l=[[0,0]]
            rectified_left_points_result.append(points_l)
          else:

            points_l=[[point[0]+max(0,int(bbox[0]-.05*(bbox[2]-bbox[0]))),point[1]+max(0,int(bbox[1]-.05*(bbox[3]-bbox[1])))]]

            rectified_left_points_result.append(points_l)








    #doing random data augmentation for the right rectified images

    # rectified_left_points_xlist=None
    # rectified_left_points_ylist=None

    # imgl_rectified = cv2.imread('tmp_l.jpg')

    # for h in range(50):
    #   # Step 1: Load the image

    #   height, width = imgl_rectified.shape[:2]

    #   imgl_temp = imgl_rectified.copy()
    #   # Step 2: Apply random augmentation
    #   # Randomly choose an augmentation: noise, stretching, or cropping
    #   augmentation = random.choice(['noise', 'stretch','crop'])
    #   # augmentation = random.choice(['noise', 'stretch'])

    #   # Initialize transformation matrix
    #   M = np.eye(3)

    #   if augmentation == 'noise':
    #       # Check the number of channels in the image
    #       if len(imgl_temp.shape) == 3:
    #           # If the image is color (BGR), create noise with 3 channels
    #           noise = np.random.randint(0, 50, imgl_temp.shape, dtype='uint8')
    #       else:
    #           # If the image is grayscale, create single channel noise
    #           noise = np.random.randint(0, 50, (height, width), dtype='uint8')

    #       imgl_temp = cv2.add(imgl_temp, noise)

    #   elif augmentation == 'stretch':
    #       # Randomly choose stretch factors
    #       fx, fy = random.uniform(0.8, 1.2), random.uniform(0.8, 1.2)
    #       imgl_temp = cv2.resize(imgl_temp, None, fx=fx, fy=fy)
    #       M = np.array([[fx, 0, 0], [0, fy, 0], [0, 0, 1]])

    #   elif augmentation == 'crop':
    #       # Randomly choose crop size
    #       x, y = random.randint(0, width // 4), random.randint(0, height // 4)
    #       w, h = width - 2*x, height - 2*y
    #       imgl_temp = imgl_temp[y:y+h, x:x+w]
    #       M = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]])

    #   # Step 3: Detect keypoints using the augmented image
    #   rectified_left_points = []

    #   for bbox in bboxes_l:
    #       # Extract and save the cropped image from the bounding box
    #       cropped_img = imgl_temp[max(0, int(bbox[1] - 0.05 * (bbox[3] - bbox[1]))):int(min(imgl_temp.shape[0], bbox[3] + 0.05 * (bbox[3] - bbox[1]))), max(0, int(bbox[0] - 0.05 * (bbox[2] - bbox[0]))):min(imgl_temp.shape[1], int(bbox[2] + 0.05 * (bbox[2] - bbox[0])))]
    #       cv2.imwrite('box_l.jpg', cropped_img)

    #       p_l = run_thunder('box_l.jpg')


    #       for point in p_l[0]:
    #           if point[0] == point[1] == 0:
    #               points_l = [[0, 0]]
    #           else:
    #               # Adjust keypoints based on augmentation
    #               x_adj, y_adj = point[0] + max(0, int(bbox[0] - 0.05 * (bbox[2] - bbox[0]))), point[1] + max(0, int(bbox[1] - 0.05 * (bbox[3] - bbox[1])))

    #               # Transform points back according to the inverse of M
    #               if augmentation != 'noise':
    #                   inv_M = np.linalg.inv(M)
    #                   x_adj, y_adj, _ = np.dot(inv_M, [x_adj, y_adj, 1])

    #               points_l = [[int(x_adj), int(y_adj)]]

    #           rectified_left_points.append(points_l)




    #   temp_left=np.array(rectified_left_points).reshape(17,2)



    #   temp_left_x=temp_left[:,0]
    #   temp_left_y=temp_left[:,1]




    #   if rectified_left_points_xlist is None:
    #     rectified_left_points_xlist=temp_left_x
    #   else:
    #     rectified_left_points_xlist=np.vstack((rectified_left_points_xlist, temp_left_x))



    #   if rectified_left_points_ylist is None:
    #     rectified_left_points_ylist=temp_left_y
    #   else:
    #     rectified_left_points_ylist=np.vstack((rectified_left_points_ylist, temp_left_y))




    # rectified_left_points_xlist = rectified_left_points_xlist.astype(float).T
    # rectified_left_points_ylist = rectified_left_points_ylist.astype(float).T


    # #now you get 17,100

    # rectified_left_points_xlist=rectified_left_points_xlist.tolist()
    # rectified_left_points_ylist=rectified_left_points_ylist.tolist()

    # rectified_left_points_xlist_clean = [[value for value in sublist if value != 0.0] for sublist in rectified_left_points_xlist]
    # rectified_left_points_ylist_clean = [[value for value in sublist if value != 0.0] for sublist in rectified_left_points_ylist]


    # for i in range(len(rectified_left_points_xlist_clean)):
    #   rectified_left_points_xlist_clean[i]= sorted(rectified_left_points_xlist_clean[i])
    #   n=len(rectified_left_points_xlist_clean[i])
    #   lower_index = int(np.ceil(n * 0.25))
    #   upper_index = int(np.floor(n * 0.75))
    #   rectified_left_points_xlist_clean[i]=rectified_left_points_xlist_clean[i][lower_index:upper_index]


    # for i in range(len(rectified_left_points_ylist_clean)):
    #   rectified_left_points_ylist_clean[i]= sorted(rectified_left_points_ylist_clean[i])
    #   n=len(rectified_left_points_ylist_clean[i])
    #   lower_index = int(np.ceil(n * 0.25))
    #   upper_index = int(np.floor(n * 0.75))
    #   rectified_left_points_ylist_clean[i]=rectified_left_points_ylist_clean[i][lower_index:upper_index]



    # #delete the high variance ones
    # for i in range(len(rectified_left_points_xlist_clean)):
    #   # print(np.var(rectified_left_points_xlist_clean[i]))
    #   if np.var(rectified_left_points_xlist_clean[i])>variance_threshold:
    #     rectified_left_points_xlist_clean[i]=[0]

    # for i in range(len(rectified_left_points_ylist_clean)):
    #   # print(np.var(rectified_left_points_ylist_clean[i]))
    #   if np.var(rectified_left_points_ylist_clean[i])>variance_threshold:
    #     rectified_left_points_ylist_clean[i]=[0]




    # for i in range(len(rectified_left_points_xlist_clean)):
    #   # #this is the mean
    #   # i_mean=sum(rectified_left_points_xlist_clean[i])/len(rectified_left_points_xlist_clean[i])
    #   # rectified_left_points_xlist_clean[i]=i_mean
      
      
    #   #get the median:

    #   len_n=len(rectified_left_points_xlist_clean[i])
    #   if len_n>=2:
    #     if len_n%2 ==1:
    #       rectified_left_points_xlist_clean[i]=rectified_left_points_xlist_clean[i][len_n//2]
    #     else:
    #       mid1=rectified_left_points_xlist_clean[i][len_n // 2 - 1 ]
    #       mid2=rectified_left_points_xlist_clean[i][len_n // 2]
    #       rectified_left_points_xlist_clean[i]=(mid1+mid2)/2
    #   elif len_n==1:
    #     rectified_left_points_xlist_clean[i]=rectified_left_points_xlist_clean[i][0]
    #   else:
    #     #len_n=0
    #     rectified_left_points_xlist_clean[i]=0
      

    # for i in range(len(rectified_left_points_ylist_clean)):
    #   # #mean
    #   # i_mean=sum(rectified_left_points_ylist_clean[i])/len(rectified_left_points_ylist_clean[i])
    #   # rectified_left_points_ylist_clean[i]=i_mean

    #   #median:
    #   len_n=len(rectified_left_points_ylist_clean[i])
    #   if len_n>=2:

    #     if len_n%2 ==1:
    #       rectified_left_points_ylist_clean[i]=rectified_left_points_ylist_clean[i][len_n//2]
    #     else:
    #       mid1=rectified_left_points_ylist_clean[i][len_n // 2 - 1 ]
    #       mid2=rectified_left_points_ylist_clean[i][len_n // 2]
    #       rectified_left_points_ylist_clean[i]=(mid1+mid2)/2
    #   elif len_n==1:
    #      rectified_left_points_ylist_clean[i]=rectified_left_points_ylist_clean[i][0]
    #   else:
    #     #len_n=0
    #     rectified_left_points_ylist_clean[i]=0


    # result_left=[]


    # assert len(rectified_left_points_xlist_clean)==len(rectified_left_points_ylist_clean)
    # for i in range(len(rectified_left_points_xlist_clean)):
    #   result_left.append([[rectified_left_points_xlist_clean[i],rectified_left_points_ylist_clean[i]]])


    # rectified_left_points_result=result_left
    # #end of left





    # #ssim


    # adjusted_bboxes_r = [adjust_bbox(bbox) for bbox in bboxes_r]
    # adjusted_bboxes_l = [adjust_bbox(bbox) for bbox in bboxes_l]
    # print(adjusted_bboxes_r, adjusted_bboxes_l)  # Print the adjusted bounding boxes

    


    person_r=rectified_right_points_result


    person_l=rectified_left_points_result

    # print(len(person_r),len(person_l))

    assert len(person_r)==len(person_l)

    num_points=len(person_r)

    # scale_factor=4



    for i in range(num_points):

      im_l_float32=imgl_rectified.astype('float32')/255.0
      im_r_float32=imgr_rectified.astype('float32')/255.0

      # new_width = im_r_float32.shape[1] // scale_factor
      # new_height = im_r_float32.shape[0] // scale_factor

      # resized_im_r = cv2.resize(im_r_float32, (new_width, new_height), interpolation=cv2.INTER_AREA)
      # resized_im_l = cv2.resize(im_l_float32, (new_width, new_height), interpolation=cv2.INTER_AREA)





      #so for the ssim, it have to be the situation where points r and points l are not 0
      if(person_r[i][0][0]!=0 and person_r[i][0][1]!=0 and person_l[i][0][0]!=0 and person_l[i][0][1]!=0):
        # print(person_r[i][0][0:2],person_l[i][0][0:2])

        
        # resize_person_r=[person_r[i][0][0]/scale_factor,person_r[i][0][1]/scale_factor]
        # resize_person_l=[person_l[i][0][0]/scale_factor,person_l[i][0][1]/scale_factor]

        # print(resize_person_r, resize_person_l)

        p2=sweep_line_block(im_r_float32, im_l_float32 , person_r[i][0][0:2],person_l[i][0][0:2],100,100, shift)

        # p2=sweep_line_block(resized_im_r , resized_im_l , resize_person_r,resize_person_l,int(200/scale_factor),int(200/scale_factor), int(shift/scale_factor))
        

        # orgin_p2=[p2[0]*scale_factor, p2[1]*scale_factor]

        # rectified_left_points_result[i]=[orgin_p2]
        
        rectified_left_points_result[i]=[p2]





    print('---------------------------')
    print(rectified_left_points_result)
    print(rectified_right_points_result)
    print('---------------------------')


    #352
    #frame0
    #left 474,707  644,735 right 714 704   897,733
    #25

    #left 679,738  883,772 right 935,738   1154,772
    #27


    #left foot
    #left 315,839 407 839  right532 839 631 839
    #got 13

    #right foot
    #left 174,742 222,844  right 383,742 429 844
    #got 16

    #left aram
    #covered

    #right leg
    #267,484 308,634 491,484 529,634
    #21.44


    #frame1
    #left foot 
    #315,839 407,839 532,839 631,839
    #got 16

    #right foot
    #231,766 288,850 444,766 503,850
    #got 15.8

    #right aram
    #199,214 150,338 439,214 380,338
    #19.38

    #right leg
    #
    #260,486 312,635 494,486 532,635
    #24.95

    #frame2
    #left foot
    #315,805 407,840 534,805 632,840

    #right foot
    #469,857 469,929 693,857 685,929

    #right aram
    #229,217 181,343 466,217 406,343
    #20.77


    #367
    #frame0
    #right aram
    #120,206 36,210 358,206 263,206
    #9

    #left aram
    #279,237 321,334 528,237 565,334
    #15.8



    #frame8
    #left aram
    #616,248 654,356 886,248 918,356
    #15.07

    #right leg
    #519,487 562,654 754,482 809,654
    #49.86

    #frame12
    #left aram
    #730,230 775,334 1008,230 1050,334
    #15.06

    #right leg
    #  603,489 611,687 855,489 855,687 
    # 56.14

    



    
    ##260,486 312,635 494,486 532,635
    rectified_left_points_result=[[[260,486]],[[312,635]]]
    rectified_right_points_result=[[[494,486]],[[532,635]]]


    print('---------------------------')
    print(rectified_left_points_result)
    print(rectified_right_points_result)
    print('---------------------------')


    # #                            0                   1                    2                   3                           4               5                 6     
    # rectified_left_points_result=[[[271, 147]],     [[0, 0]],          [[0, 0]],          [[0, 0]],             [[0, 0]],         [[317, 229]],     [[190, 218]],     [[0, 0]],         [[130, 332]],     [[367, 334]],     [[258, 364]],     [[328, 463]],     [[252, 466]],     [[347, 640]],     [[278, 646]],     [[333, 799]],     [[191, 752]]]
    # rectified_right_points_result=[[[519.5, 145.5]], [[530.0, 130.0]], [[503.5, 132.5]], [[534.5, 137.0]],      [[470.5, 143.0]], [[561.5, 229.0]], [[436.0, 218.0]], [[527.5, 342.0]], [[361.5, 332.0]], [[608.5, 333.0]], [[484.5, 363.0]], [[563.0, 462.0]], [[480.0, 466.0]], [[576.0, 639.0]], [[503.0, 645.0]], [[553.0, 797.5]], [[401.0, 751.0]]]

    if verbose:
        print(f"YOLO DONE, {len(rectified_right_points_result)} people found")

        # for pr in rectified_right_points_result:
        #     center=[round(pr[0][0]),round(pr[0][1])]
        #     cv2.circle(imgr_rectified, center, 1, (0, 0, 255), -1)
        cv2.imwrite('./352_results/right/test_rectified_r_{}.jpg'.format(count),imgr_rectified)


    if verbose:
        print(f"YOLO DONE, {len(rectified_left_points_result)} people found")
        # for pl in rectified_left_points_result:
        #     center=[round(pl[0][0]),round(pl[0][1])]
        #     cv2.circle(imgl_rectified, center, 1, (0, 255, 0), -1)
        cv2.imwrite('./352_results/left/test_rectified_l_{}.jpg'.format(count),imgl_rectified)





    #******************* get 3D points and triangulation


    # Remap left and right points to original images using H1 and H2 respectively
    left_points=[]
    right_points=[]
    for i in range (len(rectified_left_points_result)):
        left=[]
        right=[]
        for j in range(len(rectified_left_points_result[i])):
            if ((rectified_left_points_result[i][j][0]==0 and rectified_left_points_result[i][j][1]==0) or (rectified_right_points_result[i][j][0]==0 and rectified_right_points_result[i][j][1]==0)):
              left=np.array([[0,0]])
              right=np.array([[0,0]])
            else:
              rpl=np.hstack((np.array(rectified_left_points_result[i][j])[0:2].reshape((1,2)),np.array([[1]]))).reshape((3,1))
              pl=np.linalg.inv(H1).dot(rpl)[0:2]/np.linalg.inv(H1).dot(rpl)[2]
              rpr=np.hstack((np.array(rectified_right_points_result[i][j])[0:2].reshape((1,2)),np.array([[1]]))).reshape((3,1))
              pr=np.linalg.inv(H2).dot(rpr)[0:2]/np.linalg.inv(H2).dot(rpr)[2]
              left.append(pl[0:2].reshape((1,2)))
              right.append(pr[0:2].reshape((1,2)))

        left_points.append(left)
        right_points.append(right)

    if verbose:
        print(f"Original points remapped")

    points=[]
    for  uvs_l,uvs_r in zip(left_points,right_points):
        # print((uvs_l, uvs_r))
        # if ((uvs_l[0][0]==0 and uvs_l[0][1]==0) or (uvs_r[0][0]==0 and uvs_r[0][1]==0)):
        # if ((uvs_l==[[0,0]]) or (uvs_r==[[0,0]])):
        if (np.any(uvs_l == [0,0]) or np.any(uvs_r == [0,0])):

          points.append([np.array([[0],[0],[0]])])
        else:
          points.append(triangulate_points(mtxl,mtxr,R,T,[uvs_l],[uvs_r]))



    vec=[]

    points=np.array(points).reshape(2,3)

    print('3D points:',points)

    # vec_inds=[[6,5],[6,8],[8,10],[5,7],[7,9],[12,14],[14,16],[11,13],[13,15],[6,12],[5,11],[11,12],[0,6],[0,5],[5,12],[6,11]]
    # vec_inds=[[6,5],[6,8],[8,10],[5,7],[7,9],[12,14],[14,16],[11,13],[13,15],[6,12],[5,11],[11,12]]    
    vec_inds=[[0,1]]
    for pair in vec_inds:
      # if (pair[0]==0 or pair[1]==0):
      #   vec.append(0)

      # else:
      result=np.linalg.norm(points[pair[0]].reshape(3,1)-points[pair[1]].reshape(3,1))/10.0

      vec.append(result)



    return vec



# def normalize_rgb_channels(image, scale=150):
    
#     float_img = image.astype(np.float32)
#     print(float_img.shape)
#     max_rgb = np.max(float_img, axis=2, keepdims=True)
#     # print(max_rgb)
#     print(max_rgb.shape)

#     max_rgb[max_rgb == 0] = 100
    
#     normalized_img = (float_img / max_rgb) * scale
    
#     # normalized_img = np.clip(normalized_img, 0, 255).astype(np.uint8)

#     return normalized_img




def main():


  # Pull required data from settings file (for now, added manually)

  # Left intrinsic matrix
  ## Can be optimized with box/laser/caliper or simply use matrix from initial single cam calibration
  mtxl=np.array([[1.39375809e+03, 0.00000000e+00, 9.51204303e+02],
                [0.00000000e+00, 1.38294458e+03, 5.05242937e+02],
                [0.0,0.0,1.0]],dtype='float32')


  ## Right intrinsic matrix
  ### Can be optimized with box/laser/caliper or simply use matrix from initial single cam calibration
  mtxr=np.array([[1.33775214e+03, 0.00000000e+00, 9.91418067e+02],
                [0.00000000e+00,1.32903932e+03, 5.26007380e+02],
                [0.0,0.0,1.0]],dtype='float32')


  ## Rotation extrinsic matrix
  R=np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])

  ## Translation extrinsic matrix
  T=np.array([[65.0],[0.0],[0.0]])

  ## Left rectification homography
  ### Can be computed with VIPER-REGISTRAR/lib/camera_calibration/rectification_point_method.py


  H1=np.array(
  [[1.06266554e-01,  7.98644097e-03, -2.21196882e+01],
  [ -3.21850014e-03,1.19391917e-01,  8.35946228e-01],
  [ -2.94622121e-06,  1.98558892e-06, 1.19610744e-01]])


  H2=np.array(
    [[ 1.02445672e+00,  1.27651110e-02, -3.03716088e+01],
  [ 1.34116162e-03,1.00009434e+00, -1.33845810e+00],
  [2.55566031e-05,  3.18444762e-07,9.75293701e-01]]
  )



  cap1 = cv2.VideoCapture('./352.mp4')
  cap2 = cv2.VideoCapture('./352_stereo.mp4')

  if not cap1.isOpened() or not cap2.isOpened():
      print("Error opening video streams or files")

  fps1 = cap1.get(cv2.CAP_PROP_FPS)
  fps2 = cap2.get(cv2.CAP_PROP_FPS)

  assert fps1==fps2

  fourcc = cv2.VideoWriter_fourcc(*'MP4V')


  left_points = []
  right_points = []

  count=0
  vecs={}

  # while cap1.isOpened() and cap2.isOpened():
  for i in range(1):

    # ret1, frame1 = cap1.read()
    # ret2, frame2 = cap2.read()

    # if not ret1 or not ret2:
    #     # Break the loop if there are no frames left to read
    #     break

    frame1=cv2.imread('./352_results/left/test_rectified_l_1.jpg')
    frame2=cv2.imread('./352_results/right/test_rectified_r_1.jpg')

    # normalized_image1 = normalize_rgb_channels(frame1)
    # normalized_image2 = normalize_rgb_channels(frame2)


    # cv2.imwrite('./normalized_image1.jpg', normalized_image1)
    # cv2.imwrite('./normalized_image2.jpg', normalized_image2)



    vec=person_vecs_identification(count,mtxl,mtxr,R,T,H1,H2,frame1, frame2,verbose=True,shift=20)
    

    count+=1


    print(vec)




if __name__ == "__main__":
    main()