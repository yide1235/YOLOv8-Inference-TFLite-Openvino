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






def person_vecs_identification(count,imgl,imgr,verbose=False, shift=30):


    # imgl_rectified = cv2.warpPerspective(imgl, H1, (1920,1080))
    # imgr_rectified = cv2.warpPerspective(imgr, H2, (1920,1080))
    # imgl_rectified=imgr
    imgr_rectified=imgl

    #only use for ssim
    # im_l_float32=imgl_rectified.astype('float32')/255.0
    im_r_float32=imgr_rectified.astype('float32')/255.0
    #only use for ssim

    cv2.imwrite('tmp_r.jpg',imgr_rectified)
    # cv2.imwrite('tmp_l.jpg',imgl_rectified)
    # cv2.imwrite('tmp_save.jpg',imgr_rectified) #here looks good

    bboxes_r=run_yolo(['tmp_r.jpg'])
    # bboxes_l=run_yolo(['tmp_l.jpg'])

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




    print('---------------------------')

    print(rectified_right_points_result)
    print('---------------------------')


    if verbose:
        print(f"YOLO DONE, {len(rectified_right_points_result)} people found")

        for pr in rectified_right_points_result:
            center=[round(pr[0][0]),round(pr[0][1])]
            cv2.circle(imgr_rectified, center, 5, (0, 255, 0), -1)
        cv2.imwrite('./6733_{}.jpg'.format(count),imgr_rectified)






    return vec





def main():








  left_points = []
  right_points = []

  count=0
  vecs={}

  # while cap1.isOpened() and cap2.isOpened():
  for i in range(1):


    frame1=cv2.imread('./IMG_6733.jpg')
    frame2=cv2.imread('./352/right.jpg')


    vec=person_vecs_identification(count, frame1, frame2,verbose=True,shift=20)


    # frame1=cv2.imread('./IMG_67231.jpg')
    # frame2=cv2.imread('./352/right.jpg')


    # vec=person_vecs_identification(count, frame1, frame2,verbose=True,shift=20)

    # frame1=cv2.imread('./IMG_6733.jpg')
    # frame2=cv2.imread('./352/right.jpg')


    # vec=person_vecs_identification(count, frame1, frame2,verbose=True,shift=20)
    

    count+=1


    print(vec)




if __name__ == "__main__":
    main()





