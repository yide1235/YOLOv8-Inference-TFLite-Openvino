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






def person_vecs_identification(count, mtxl,mtxr,R,T,H1,H2,imgl,imgr,verbose=False, shift=15):


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



    # this is for without the random augmentation
    # for g in range(2):
    #   # Get pose points in right image
    #   ## Will need to manage according to model (yolov8?) and what type of output is received. May need to save extra info (like confidences and bounding boxes) to add back later

    #   # rectified_right_points=run_yolo_pose(['tmp.jpg'])

    #   rectified_right_points=[]

    #   # print('box: ', bboxes_r)
    #   for bbox in bboxes_r:


    #       cv2.imwrite('box_r.jpg',imgr_rectified[max(0,int(bbox[1]-.05*(bbox[3]-bbox[1]))):int(min(imgr_rectified.shape[0],bbox[3]+.05*(bbox[3]-bbox[1]))),max(0,int(bbox[0]-.05*(bbox[2]-bbox[0]))):min(imgr_rectified.shape[1],int(bbox[2]+.05*(bbox[2]-bbox[0])))])

    #       p_r=run_thunder('box_r.jpg')

    #       for point in p_r[0]:
    #         if point[0]==point[1]==0:
    #           points_r=[[0,0]]
    #           rectified_right_points.append(points_r)
    #         else:
    #           points_r=[[point[0]+max(0,int(bbox[0]-.05*(bbox[2]-bbox[0]))),point[1]+max(0,int(bbox[1]-.05*(bbox[3]-bbox[1])))] ]

    #           rectified_right_points.append(points_r)s




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



    for i in range(len(rectified_right_points_xlist_clean)):
      #this is the mean
      # i_mean=sum(rectified_right_points_xlist_clean[i])/len(rectified_right_points_xlist_clean[i])
      # rectified_right_points_xlist_clean[i]=i_mean
      #get the median:

      len_n=len(rectified_right_points_xlist_clean[i])
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
      #mean
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

    #doing random data augmentation for the right rectified images

    rectified_left_points_xlist=None
    rectified_left_points_ylist=None

    imgl_rectified = cv2.imread('tmp_l.jpg')

    for h in range(50):
      # Step 1: Load the image

      height, width = imgl_rectified.shape[:2]

      imgl_temp = imgl_rectified.copy()
      # Step 2: Apply random augmentation
      # Randomly choose an augmentation: noise, stretching, or cropping
      augmentation = random.choice(['noise', 'stretch','crop'])
      # augmentation = random.choice(['noise', 'stretch'])

      # Initialize transformation matrix
      M = np.eye(3)

      if augmentation == 'noise':
          # Check the number of channels in the image
          if len(imgl_temp.shape) == 3:
              # If the image is color (BGR), create noise with 3 channels
              noise = np.random.randint(0, 50, imgl_temp.shape, dtype='uint8')
          else:
              # If the image is grayscale, create single channel noise
              noise = np.random.randint(0, 50, (height, width), dtype='uint8')

          imgl_temp = cv2.add(imgl_temp, noise)

      elif augmentation == 'stretch':
          # Randomly choose stretch factors
          fx, fy = random.uniform(0.8, 1.2), random.uniform(0.8, 1.2)
          imgl_temp = cv2.resize(imgl_temp, None, fx=fx, fy=fy)
          M = np.array([[fx, 0, 0], [0, fy, 0], [0, 0, 1]])

      elif augmentation == 'crop':
          # Randomly choose crop size
          x, y = random.randint(0, width // 4), random.randint(0, height // 4)
          w, h = width - 2*x, height - 2*y
          imgl_temp = imgl_temp[y:y+h, x:x+w]
          M = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]])

      # Step 3: Detect keypoints using the augmented image
      rectified_left_points = []

      for bbox in bboxes_l:
          # Extract and save the cropped image from the bounding box
          cropped_img = imgl_temp[max(0, int(bbox[1] - 0.05 * (bbox[3] - bbox[1]))):int(min(imgl_temp.shape[0], bbox[3] + 0.05 * (bbox[3] - bbox[1]))), max(0, int(bbox[0] - 0.05 * (bbox[2] - bbox[0]))):min(imgl_temp.shape[1], int(bbox[2] + 0.05 * (bbox[2] - bbox[0])))]
          cv2.imwrite('box_l.jpg', cropped_img)

          p_l = run_thunder('box_l.jpg')


          for point in p_l[0]:
              if point[0] == point[1] == 0:
                  points_l = [[0, 0]]
              else:
                  # Adjust keypoints based on augmentation
                  x_adj, y_adj = point[0] + max(0, int(bbox[0] - 0.05 * (bbox[2] - bbox[0]))), point[1] + max(0, int(bbox[1] - 0.05 * (bbox[3] - bbox[1])))

                  # Transform points back according to the inverse of M
                  if augmentation != 'noise':
                      inv_M = np.linalg.inv(M)
                      x_adj, y_adj, _ = np.dot(inv_M, [x_adj, y_adj, 1])

                  points_l = [[int(x_adj), int(y_adj)]]

              rectified_left_points.append(points_l)




      temp_left=np.array(rectified_left_points).reshape(17,2)



      temp_left_x=temp_left[:,0]
      temp_left_y=temp_left[:,1]




      if rectified_left_points_xlist is None:
        rectified_left_points_xlist=temp_left_x
      else:
        rectified_left_points_xlist=np.vstack((rectified_left_points_xlist, temp_left_x))



      if rectified_left_points_ylist is None:
        rectified_left_points_ylist=temp_left_y
      else:
        rectified_left_points_ylist=np.vstack((rectified_left_points_ylist, temp_left_y))




    rectified_left_points_xlist = rectified_left_points_xlist.astype(float).T
    rectified_left_points_ylist = rectified_left_points_ylist.astype(float).T


    #now you get 17,100

    rectified_left_points_xlist=rectified_left_points_xlist.tolist()
    rectified_left_points_ylist=rectified_left_points_ylist.tolist()

    rectified_left_points_xlist_clean = [[value for value in sublist if value != 0.0] for sublist in rectified_left_points_xlist]
    rectified_left_points_ylist_clean = [[value for value in sublist if value != 0.0] for sublist in rectified_left_points_ylist]


    for i in range(len(rectified_left_points_xlist_clean)):
      rectified_left_points_xlist_clean[i]= sorted(rectified_left_points_xlist_clean[i])
      n=len(rectified_left_points_xlist_clean[i])
      lower_index = int(np.ceil(n * 0.25))
      upper_index = int(np.floor(n * 0.75))
      rectified_left_points_xlist_clean[i]=rectified_left_points_xlist_clean[i][lower_index:upper_index]


    for i in range(len(rectified_left_points_ylist_clean)):
      rectified_left_points_ylist_clean[i]= sorted(rectified_left_points_ylist_clean[i])
      n=len(rectified_left_points_ylist_clean[i])
      lower_index = int(np.ceil(n * 0.25))
      upper_index = int(np.floor(n * 0.75))
      rectified_left_points_ylist_clean[i]=rectified_left_points_ylist_clean[i][lower_index:upper_index]



    for i in range(len(rectified_left_points_xlist_clean)):
      #this is the mean
      # i_mean=sum(rectified_left_points_xlist_clean[i])/len(rectified_left_points_xlist_clean[i])
      # rectified_left_points_xlist_clean[i]=i_mean
      #get the median:

      len_n=len(rectified_left_points_xlist_clean[i])
      if len_n>=2:
        if len_n%2 ==1:
          rectified_left_points_xlist_clean[i]=rectified_left_points_xlist_clean[i][len_n//2]
        else:
          mid1=rectified_left_points_xlist_clean[i][len_n // 2 - 1 ]
          mid2=rectified_left_points_xlist_clean[i][len_n // 2]
          rectified_left_points_xlist_clean[i]=(mid1+mid2)/2
      elif len_n==1:
        rectified_left_points_xlist_clean[i]=rectified_left_points_xlist_clean[i][0]
      else:
        #len_n=0
        rectified_left_points_xlist_clean[i]=0
      

    for i in range(len(rectified_left_points_ylist_clean)):
      #mean
      # i_mean=sum(rectified_left_points_ylist_clean[i])/len(rectified_left_points_ylist_clean[i])
      # rectified_left_points_ylist_clean[i]=i_mean

      #median:
      len_n=len(rectified_left_points_ylist_clean[i])
      if len_n>=2:

        if len_n%2 ==1:
          rectified_left_points_ylist_clean[i]=rectified_left_points_ylist_clean[i][len_n//2]
        else:
          mid1=rectified_left_points_ylist_clean[i][len_n // 2 - 1 ]
          mid2=rectified_left_points_ylist_clean[i][len_n // 2]
          rectified_left_points_ylist_clean[i]=(mid1+mid2)/2
      elif len_n==1:
         rectified_left_points_ylist_clean[i]=rectified_left_points_ylist_clean[i][0]
      else:
        #len_n=0
        rectified_left_points_ylist_clean[i]=0


    result_left=[]


    assert len(rectified_left_points_xlist_clean)==len(rectified_left_points_ylist_clean)
    for i in range(len(rectified_left_points_xlist_clean)):
      result_left.append([[rectified_left_points_xlist_clean[i],rectified_left_points_ylist_clean[i]]])


    rectified_left_points_result=result_left
    #end of left









    # # normal finding left points
    # # Loop through points, align corresponding left and  lines from rectified images, and get left points
    # rectified_left_points_result=[]
    # # print('box: ', bboxes_l)
    # for bbox in bboxes_l:


    #     cv2.imwrite('box_l.jpg',imgl_rectified[max(0,int(bbox[1]-.05*(bbox[3]-bbox[1]))):int(min(imgl_rectified.shape[0],bbox[3]+.05*(bbox[3]-bbox[1]))),max(0,int(bbox[0]-.05*(bbox[2]-bbox[0]))):min(imgl_rectified.shape[1],int(bbox[2]+.05*(bbox[2]-bbox[0])))])

    #     p_l=run_thunder('box_l.jpg')

    #     for point in p_l[0]:
    #       if point[0]==point[1]==0:
    #         points_l=[[0,0]]
    #         rectified_left_points_result.append(points_l)
    #       else:

    #         points_l=[[point[0]+max(0,int(bbox[0]-.05*(bbox[2]-bbox[0]))),point[1]+max(0,int(bbox[1]-.05*(bbox[3]-bbox[1])))]]

    #         rectified_left_points_result.append(points_l)








    # #ssim


    # person_r=rectified_right_points_result


    # person_l=rectified_left_points_result

    # # print(len(person_r),len(person_l))

    # assert len(person_r)==len(person_l)

    # num_points=len(person_r)

    # for i in range(num_points):

    #   im_l_float32=imgl_rectified.astype('float32')/255.0
    #   im_r_float32=imgr_rectified.astype('float32')/255.0


    #   #so for the ssim, it have to be the situation where points r and points l are not 0
    #   if(person_r[i][0][0]!=0 and person_r[i][0][1]!=0 and person_l[i][0][0]!=0 and person_l[i][0][1]!=0):
    #     print(person_r[i][0][0:2],person_l[i][0][0:2])

    #     p2=sweep_line_block(im_r_float32, im_l_float32 , person_r[i][0][0:2],person_l[i][0][0:2],20,20, shift)

    #     rectified_left_points_result[i]=[p2]








    # aligh the height
    assert len(rectified_left_points_result)==len(rectified_right_points_result)

    for i in range(len(rectified_left_points_result)):
      rectified_left_points_result[i][0][1]=rectified_right_points_result[i][0][1]










    print('---------------------------')
    print(rectified_left_points_result)
    print(rectified_right_points_result)
    print('---------------------------')


    if verbose:
        print(f"YOLO DONE, {len(rectified_right_points_result)} people found")

        for pr in rectified_right_points_result:
            center=[int(pr[0][0]),int(pr[0][1])]
            cv2.circle(imgr_rectified, center, 5, (0, 0, 255), -1)
        cv2.imwrite('./366_results/right/test_rectified_r_{}.jpg'.format(count),imgr_rectified)


    if verbose:
        print(f"YOLO DONE, {len(rectified_left_points_result)} people found")
        for pl in rectified_left_points_result:
            center=[int(pl[0][0]),int(pl[0][1])]
            cv2.circle(imgl_rectified, center, 5, (0, 255, 0), -1)
        cv2.imwrite('./366_results/left/test_rectified_l_{}.jpg'.format(count),imgl_rectified)





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

    points=np.array(points).reshape(17,3)
    # vec_inds=[[5,6],[5,7],[5,11],[6,8],[6,12],[7,9],[8,10],[11,12],[11,13],[12,14],[13,15],[14,16]]
    vec_inds=[[6,5],[6,8],[8,10],[5,7],[7,9],[12,14],[14,16],[11,13],[13,15]]
    for pair in vec_inds:
        if (pair[0]==0 or pair[1]==0):
          vec.append(0)

        else:
          result=np.linalg.norm(points[pair[0]].reshape(3,1)-points[pair[1]].reshape(3,1))/10.0

          vec.append(result)



    return vec




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



  cap1 = cv2.VideoCapture('./366.mp4')
  cap2 = cv2.VideoCapture('./366_stereo.mp4')

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

  while cap1.isOpened() and cap2.isOpened():
  # for i in range(2):

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()


    if not ret1 or not ret2:
        # Break the loop if there are no frames left to read
        break

    vec=person_vecs_identification(count,mtxl,mtxr,R,T,H1,H2,frame1,frame2,verbose=True,shift=10)
    



    count+=1


    print(vec)

    for i in range(len(vec)):
      # vecs[i].append(vec[i])
      if vec[i]!=0: #only add non 0 items
        if i in vecs:
            vecs[i] = np.append(vecs[i], vec[i])
        else:
            vecs[i]=np.array(vec[i])

  #this is end of loop


  print(vecs)


  delete_threshold=10

  for key, value in vecs.items():

    sorted_vecs=np.sort(value)
    sorted_vecs=sorted_vecs[delete_threshold:-delete_threshold]
    # vecs[key]=np.mean(sorted_vecs)
    vecs[key]=np.median(sorted_vecs)

  print(vecs)


  ##--------format is [[6,5],[6,8],[8,10],[5,7],[7,9],[12,14],[14,16],[11,13],[13,15]]





if __name__ == "__main__":
    main()
