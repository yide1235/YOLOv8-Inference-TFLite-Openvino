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

    rectified_left_points_result=[[[856.0,132.0]],[[1207.0,178.0]]]
    rectified_right_points_result=[[[837.0,132.0]],[[1218.0,178.0]]]


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


    #triangulation
    # #mine triangulation
    # points=[]
    # assert len(rectified_right_points_result)==len(rectified_left_points_result)
    # for i in range(len(rectified_right_points_result)):
    #   homo_point=cv2.triangulatePoints(P1,P2, np.array(rectified_left_points_result[i][0]),np.array(rectified_right_points_result[i][0]))
    #   euclidean_point = homo_point[:3] / homo_point[3]
    #   points.append(euclidean_point)

    # print(points)


    # vec=[]

    # #[0,1 should be 20x9, 0,2 should be 20x8]
    # # vec_inds=[[0,1],[0,2]]
    # vec_inds=[[0,1]]
    # for pair in vec_inds:

    #   result=np.linalg.norm(points[pair[0]].reshape(3,1)-points[pair[1]].reshape(3,1))/10.0

    #   vec.append(result)

    # print(vec)

    # return vec


    #--------------------------------------
    #sgbm
    imgl_rectified_gray=cv2.cvtColor(imgl_rectified, cv2.COLOR_BGR2GRAY)
    imgr_rectified_gray=cv2.cvtColor(imgr_rectified, cv2.COLOR_BGR2GRAY)


    mindisparity = 32
    SADWindowSize = 16
    ndisparities = 176

    P1 = 4 * 1 * SADWindowSize * SADWindowSize
    P2 = 32 * 1 * SADWindowSize * SADWindowSize


    sgbm = cv2.StereoSGBM_create(mindisparity, ndisparities, SADWindowSize)
    sgbm.setP1(P1)
    sgbm.setP2(P2)

    sgbm.setPreFilterCap(60)
    sgbm.setUniquenessRatio(30)
    sgbm.setSpeckleRange(2)
    sgbm.setSpeckleWindowSize(200)
    sgbm.setDisp12MaxDiff(1)
    disp = sgbm.compute(imgl_rectified_gray, imgr_rectified_gray)


    xyz = cv2.reprojectImageTo3D(disp, Q, handleMissingValues=True)
    xyz = xyz * 16


    disp = disp.astype(np.float32) / 16.0
    disp8U = cv2.normalize(disp, disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disp8U = cv2.medianBlur(disp8U, 9)



    def onMouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print('point (%d, %d) 3d (%f, %f, %f)' % (x, y, xyz[y, x, 0], xyz[y, x, 1], xyz[y, x, 2]))



    cv2.imshow("disparity", disp8U)
    cv2.setMouseCallback("disparity", onMouse, 0)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


    return None







def main():


  # Pull required data from settings file (for now, added manually)

  # Left intrinsic matrix
  ## Can be optimized with box/laser/caliper or simply use matrix from initial single cam calibration
  mtxl=np.array([[1.478511681180142e+03,-14.218342449624647,9.464932100423684e+02],
                [0,1.483404540149269e+03,5.715335105009643e+02],
                [0.0,0.0,1.0]],dtype='float32')


  ## Right intrinsic matrix
  ### Can be optimized with box/laser/caliper or simply use matrix from initial single cam calibration
  mtxr=np.array([[1.483725234734510e+03,-9.155533998701358,1.008339319700266e+03],
                [0,1.488160735091025e+03,5.689316228334426e+02],
                [0.0,0.0,1.0]],dtype='float32')


  H1=np.array(
  [[1.012072003088034,0.005186054898182,1.295070718572306e-05],
  [ 0.014279034729042,0.998223715466015,2.723184838592885e-06],
  [ -48.090041593466680,-10.942180163631065,0.985995083176619]])


  H2=np.array(
    [[ 1.002812210203788,0.001090109147710,6.740328138703862e-06],
  [ 0.006522681649863,0.992047562671343,-2.526104893078166e-06],
  [-83.217490859349000,10.383912879891454,0.994583333601249]]
  )



  ## Rotation extrinsic matrix
  # R=np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
  R=np.array([[0.999957998490265,5.289684440081520e-04,0.009149942498662],
  [-5.992744335342327e-04,0.999970307834566,0.007682728642718],
  [-0.009145606896039,-0.007687889283124,0.999928624569211]])

  ## Translation extrinsic matrix
  # T=np.array([[65.0],[0.0],[0.0]])
  T=np.array([[-65.969645462311490],[-0.184823062330663],[0.659080466937241]])


  #************opencv requires distortion order is k1 k2 p1 p2 k3 
  # in opencv k is the radial, p is the tang: Distortioncoefficients=(k1 k2 p1 p2 k3)
  # Dist_l=np.array([-0.307174691028780,0.105438148557666,-0.001625174676963,5.934624077322642e-04,-0.022379781442631])
  Dist_l=np.array([0.140368188685162,-0.493614646393617,0.006302914031479,-0.012012074311655,0.368244072542426])

  Dist_r=np.array([0.148562266886875,-0.428966804979461,0.009945386535980,0.021783569329242,0.131296762274824])



  left_points = []
  right_points = []

  count=0
  vecs={}

  # while cap1.isOpened() and cap2.isOpened():
  for i in range(1):


    # frame1=cv2.imread('./left_calibration_images/frame00003.jpg')
    # frame2=cv2.imread('./right_calibration_images/frame00003.jpg')



    vec=person_vecs_identification(count,mtxl,mtxr,Dist_l,Dist_r,R,T,H1,H2,frame1, frame2,verbose=True,shift=20)
    

    count+=1






if __name__ == "__main__":
    main()
