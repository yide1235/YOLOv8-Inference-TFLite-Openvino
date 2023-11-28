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



def person_vecs_identification(count, mtxl,mtxr,Dist_l, Dist_r,R,T,H1,H2,imgl,imgr,verbose=False, shift=30):

    # imgl_rectified = cv2.warpPerspective(imgl, H1, (1080,1920))
    # imgr_rectified = cv2.warpPerspective(imgr, H2, (1920,1080))

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(mtxl, Dist_l, mtxr, Dist_r, (imgl.shape[1], imgr.shape[0]), R, T, )

    map1_x, map1_y = cv2.initUndistortRectifyMap(mtxl, Dist_l, R1, P1, (imgl.shape[1], imgl.shape[0]), cv2.CV_32FC1)
    map2_x, map2_y = cv2.initUndistortRectifyMap(mtxr, Dist_r, R2, P2, (imgr.shape[1], imgr.shape[0]), cv2.CV_32FC1)



    # imgl_rectified = cv2.remap(imgl, map1_x, map1_y , cv2.INTER_CUBIC)
    # imgr_rectified = cv2.remap(imgr, map2_x, map2_y , cv2.INTER_CUBIC)



    imgl_rectified = imgl
    imgr_rectified = imgr



    #1.8 if shift 5 pixel for width 
    #frame 00003



    # rectified_left_points_result=[[[763.0,123.0]],[[1669.0,158.0]],[[746.0,940.0]]]
    # rectified_right_points_result=[[[461.0,123.0]],[[1371.0,158.0]],[[443.0,940.0]]]



    # rectified_left_points_result=[[[856.0,132.0]],[[1207.0,178.0]]]
    # rectified_right_points_result=[[[837.0,132.0]],[[1218.0,178.0]]]



    rectified_left_points_result=[[[1241.0,83.0]],[[1579.0,99.0]]]
    rectified_right_points_result=[[[461.0,123.0]],[[1371.0,158.0]]]



    print('---------------------------')
    print(rectified_left_points_result)
    print(rectified_right_points_result)
    print('---------------------------')


 
    if verbose:
        print(f"YOLO DONE, {len(rectified_right_points_result)} people found")

        for pr in rectified_right_points_result:
            center=[round(pr[0][0]),round(pr[0][1])]
            cv2.circle(imgr_rectified, center, 3, (0, 0, 255), -1)
        cv2.imwrite('./test_rectified_r_{}.jpg'.format(count),imgr_rectified)


    if verbose:
        print(f"YOLO DONE, {len(rectified_left_points_result)} people found")
        for pl in rectified_left_points_result:
            center=[round(pl[0][0]),round(pl[0][1])]
            cv2.circle(imgl_rectified, center, 3, (0, 255, 0), -1)
        cv2.imwrite('./test_rectified_l_{}.jpg'.format(count),imgl_rectified)



    #mine triangulation
    points=[]
    assert len(rectified_right_points_result)==len(rectified_left_points_result)
    for i in range(len(rectified_right_points_result)):
      homo_point=cv2.triangulatePoints(P1,P2, np.array(rectified_left_points_result[i][0]),np.array(rectified_right_points_result[i][0]))
      euclidean_point = homo_point[:3] / homo_point[3]
      points.append(euclidean_point)

    print(points)


    vec=[]

    #[0,1 should be 20x9, 0,2 should be 20x8]
    # vec_inds=[[0,1],[0,2]]
    vec_inds=[[0,1]]
    for pair in vec_inds:

      result=np.linalg.norm(points[pair[0]].reshape(3,1)-points[pair[1]].reshape(3,1))/10.0

      vec.append(result)

    print(vec)

    return vec
    # return None






def main():


  # Pull required data from settings file (for now, added manually)

  # Left intrinsic matrix
  ## Can be optimized with box/laser/caliper or simply use matrix from initial single cam calibration
  mtxl=np.array([[1.173383946647347e+03,2.236436281663165,9.332464899020413e+02],
                [0,1.165131531930215e+03,5.439716307895008e+02],
                [0.0,0.0,1.0]],dtype='float32')


  ## Right intrinsic matrix
  ### Can be optimized with box/laser/caliper or simply use matrix from initial single cam calibration
  mtxr=np.array([[1.181746363344540e+03,2.240838730130696,9.897888035484548e+02],
                [0,1.175372124138154e+03,5.355748822826704e+02],
                [0.0,0.0,1.0]],dtype='float32')


  ## Rotation extrinsic matrix
  # R=np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
  R=np.array([[0.999674562946962,7.156083597636780e-04,-0.025500119636514],
  [-8.508972457403092e-04,0.999985619551738,-0.005294965873815],
  [0.025495963811519,0.005314940677290,0.999660796087812]])

  ## Translation extrinsic matrix
  # T=np.array([[65.0],[0.0],[0.0]])
  T=np.array([[-59.856088255692086],[0.375572239092430],[2.086220840906873]])

  H1=np.array(
  [[1.046114185096998,0.033310392723389,5.137834008025721e-05],
  [ -0.005764673909573,1.008170008566692,2.070697982561330e-06],
  [ -1.123218383644426e+02,-44.079363023743500,0.949102857062161]])


  H2=np.array(
    [[ 1.019795694486228,0.022221726364269,2.946068088945639e-05],
  [ -0.010434719884342,0.996952492183565,-2.446143351602487e-06],
  [-1.119881792123866e+02,-13.455692807235210,0.971540065195417]]
  )

  #************opencv requires distortion order is k1 k2 p1 p2 k3 
  # in opencv k is the radial, p is the tang: Distortioncoefficients=(k1 k2 p1 p2 k3)
  Dist_l=np.array([-0.307174691028780,0.105438148557666,-0.001625174676963,5.934624077322642e-04,-0.022379781442631])

  Dist_r=np.array([-0.298829795910150,0.076519341468483,-7.249619446728525e-04,-4.530786480269589e-04,-0.005879913795926])



  left_points = []
  right_points = []

  count=0
  vecs={}

  # while cap1.isOpened() and cap2.isOpened():
  for i in range(1):


    # frame1=cv2.imread('./367/left.jpg')
    # frame2=cv2.imread('./367/right.jpg')

    frame1=cv2.imread('./17830/left.jpg')
    frame2=cv2.imread('./17830/right.jpg')



    vec=person_vecs_identification(count,mtxl,mtxr,Dist_l,Dist_r,R,T,H1,H2,frame1, frame2,verbose=True,shift=20)
    

    count+=1






if __name__ == "__main__":
    main()
