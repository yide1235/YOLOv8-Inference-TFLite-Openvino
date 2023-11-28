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


    print('---------------------------')
    print(rectified_left_points_result)
    print(rectified_right_points_result)
    print('---------------------------')


 
    if verbose:
        print(f"YOLO DONE, {len(rectified_right_points_result)} people found")

        # for pr in rectified_right_points_result:
        #     center=[round(pr[0][0]),round(pr[0][1])]
        #     cv2.circle(imgr_rectified, center, 3, (0, 0, 255), -1)
        cv2.imwrite('./test_rectified_r_{}.jpg'.format(count),imgr_rectified)


    if verbose:
        print(f"YOLO DONE, {len(rectified_left_points_result)} people found")
        # for pl in rectified_left_points_result:
        #     center=[round(pl[0][0]),round(pl[0][1])]
        #     cv2.circle(imgl_rectified, center, 3, (0, 255, 0), -1)
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
  mtxl=np.array([[9.233004206240932e+02,3.035955607043972,9.146071220481753e+02],
                [0,9.156208095207031e+02,5.078657093183726e+02],
                [0.0,0.0,1.0]],dtype='float32')


  ## Right intrinsic matrix
  ### Can be optimized with box/laser/caliper or simply use matrix from initial single cam calibration
  mtxr=np.array([[9.217503170395427e+02,0.728255517249831,9.670200468686250e+02],
                [0,9.176781056727834e+02,5.068484830679096e+02],
                [0.0,0.0,1.0]],dtype='float32')


  ## Rotation extrinsic matrix
  # R=np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
  R=np.array([[0.999892431983596,-0.001586284169994,0.014581089275593],
  [0.001480766521700,0.999972661472416,0.007244559219225],
  [-0.014592182579690,-0.007222188747511,0.999867445313256]])

  ## Translation extrinsic matrix
  # T=np.array([[65.0],[0.0],[0.0]])
  T=np.array([[-62.399692579527310],[-0.945385030505363],[0.036890988503638]])

  H1=np.array(
  [[1.013154671634324,-0.008281139234788,1.649696794132742e-05],
  [ 0.017061091221872,1.008619758657930,3.973825172564627e-06],
  [ -34.902881740952420,-0.540309146429195,0.982770773964130]])


  H2=np.array(
    [[ 1.000525167876685,-0.014791457192926,6.998538133362168e-07],
  [ 0.010876895689142,1.002361101857093,-3.876602176580192e-06],
  [-58.984567335568890,16.899475598066260,1.001281543179227]]
  )

  #************opencv requires distortion order is k1 k2 p1 p2 k3 
  # in opencv k is the radial, p is the tang: Distortioncoefficients=(k1 k2 p1 p2 k3)
  # Dist_l=np.array([-0.307174691028780,0.105438148557666,-0.001625174676963,5.934624077322642e-04,-0.022379781442631])
  Dist_l=np.array([0.017194693844197,-0.014518246482613,-0.001248997960603,-0.010445916044528,-0.001118783923497])

  Dist_r=np.array([-0.003982852673866,-0.005843664880190,-0.001447703742141,0.010412730052838,0.013504013627738])



  left_points = []
  right_points = []

  count=0
  vecs={}

  # while cap1.isOpened() and cap2.isOpened():
  for i in range(1):


    # frame1=cv2.imread('./left_calibration_images/frame00003.jpg')
    # frame2=cv2.imread('./right_calibration_images/frame00003.jpg')
    frame1=cv2.imread('./16321/left.jpg')
    frame2=cv2.imread('./16321/right.jpg')



    vec=person_vecs_identification(count,mtxl,mtxr,Dist_l,Dist_r,R,T,H1,H2,frame1, frame2,verbose=True,shift=20)
    

    count+=1






if __name__ == "__main__":
    main()
