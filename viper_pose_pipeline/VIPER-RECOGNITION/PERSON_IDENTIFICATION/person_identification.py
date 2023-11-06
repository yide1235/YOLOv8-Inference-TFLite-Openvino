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

from yolov8 import run_yolo
from thunder import run_thunder
from line_matching_4 import sweep_line_block
from triangulate_point import triangulate_points


def person_vecs_identification(mtxl,mtxr,R,T,H1,H2,left_image_path,right_image_path,verbose=False):
    imgl=cv2.imread(left_image_path)
    imgr=cv2.imread(right_image_path)

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

    os.remove('tmp_r.jpg')
    os.remove('tmp_l.jpg')

    if len(bboxes_r)==0:
        print('NO PERSON IN IMAGE FOUND!')
        return []

    # Get pose points in right image
    ## Will need to manage according to model (yolov8?) and what type of output is received. May need to save extra info (like confidences and bounding boxes) to add back later

    # rectified_right_points=run_yolo_pose(['tmp.jpg'])

    rectified_right_points=[]

    print('box: ', bboxes_r)
    for bbox in bboxes_r:
        print(bbox)
        print(imgr_rectified.shape)
        cv2.imwrite('box_r.jpg',imgr_rectified[max(0,int(bbox[1]-.05*(bbox[3]-bbox[1]))):int(min(imgr_rectified.shape[0],bbox[3]+.05*(bbox[3]-bbox[1]))),max(0,int(bbox[0]-.05*(bbox[2]-bbox[0]))):min(imgr_rectified.shape[1],int(bbox[2]+.05*(bbox[2]-bbox[0])))])
        
        p_r=run_thunder('box_r.jpg')


        points_r=[[point[0]+max(0,int(bbox[0]-.05*(bbox[2]-bbox[0]))),point[1]+max(0,int(bbox[1]-.05*(bbox[3]-bbox[1])))] for point in p_r[0]]

        rectified_right_points.append(points_r)

    # os.remove('tmp.jpg')

    if len(rectified_right_points)==0:
        print('NO POINTS FOUND')
        return []


    if verbose:
        print(f"YOLO DONE, {len(rectified_right_points)} people found")
        print(rectified_right_points)
        for pr in rectified_right_points[0]:
            center=[int(pr[0]),int(pr[1])]
            cv2.circle(imgr_rectified, center, 5, (0, 255, 0), -1)
        cv2.imwrite('test_rectified_r.jpg',imgr_rectified)


    #-----------------------------------------------------do the same thing for left

    # Loop through points, align corresponding left and right lines from rectified images, and get left points
    rectified_left_points=[]
    print('box: ', bboxes_l)
    for bbox in bboxes_l:


        cv2.imwrite('box_l.jpg',imgl_rectified[max(0,int(bbox[1]-.05*(bbox[3]-bbox[1]))):int(min(imgl_rectified.shape[0],bbox[3]+.05*(bbox[3]-bbox[1]))),max(0,int(bbox[0]-.05*(bbox[2]-bbox[0]))):min(imgl_rectified.shape[1],int(bbox[2]+.05*(bbox[2]-bbox[0])))])
        
        p_l=run_thunder('box_l.jpg')

        points_l=[[point[0]+max(0,int(bbox[0]-.05*(bbox[2]-bbox[0]))),point[1]+max(0,int(bbox[1]-.05*(bbox[3]-bbox[1])))] for point in p_l[0]]

        rectified_left_points.append(points_l)

    # os.remove('tmp.jpg')

    if len(rectified_left_points)==0:
        print('NO POINTS FOUND')
        return []


    if verbose:
        print(f"YOLO DONE, {len(rectified_left_points)} people found")
        for pl in rectified_left_points[0]:
            center=[int(pl[0]),int(pl[1])]
            cv2.circle(imgl_rectified, center, 5, (0, 255, 0), -1)
        cv2.imwrite('test_rectified_l.jpg',imgl_rectified)



    os.remove('box_l.jpg')
    os.remove('box_r.jpg')
    #now the left is in rectified_left_points and rectifiec_right_points

    print('----')
    print(rectified_right_points)
    print(rectified_left_points)
    print('----')



    # for person in rectified_right_points:
    #     points=[]
    #     for point in person:
    #         y=int(point[1])
    #         # left_line=imgl_rectified[y,:]
    #         # right_line=imgr_rectified[y,:]
            
    #         # left_line=imgl_rectified[y,round(bboxes_l[0][0]):round(bboxes_l[0][2])]

    #         # right_line=imgr_rectified[y,round(bboxes_r[0][0]):round(bboxes_r[0][2])]

    #         print(point[0:2])

    #         # im_l_float32=imgl_rectified.astype('float32')/255.0
    #         # im_r_float32=imgr_rectified.astype('float32')/255.0


    #         p2=sweep_line_block(im_r_float32, im_l_float32 ,bboxes_r, bboxes_l, point[0:2],20,20)
    #         points.append(p2)

    #     rectified_left_points.append(points)

    # if verbose:
    #     print(f"Left rectified points identified")
    #     print(rectified_left_points)
    #     for pl in rectified_left_points[0]:
    #         center=[int(pl[0]),int(pl[1])]
    #         cv2.circle(imgl_rectified, center, 5, (0, 255, 0), -1)
    #     cv2.imwrite('test_rectified_l.jpg',imgl_rectified)




    # Remap left and right points to original images using H1 and H2 respectively
    left_points=[]
    right_points=[]
    for i in range (len(rectified_left_points)):
        left=[]
        right=[]
        for j in range(len(rectified_left_points[i])):
            rpl=np.hstack((np.array(rectified_left_points[i][j])[0:2].reshape((1,2)),np.array([[1]]))).reshape((3,1))
            pl=np.linalg.inv(H1).dot(rpl)[0:2]/np.linalg.inv(H1).dot(rpl)[2]
            rpr=np.hstack((np.array(rectified_right_points[i][j])[0:2].reshape((1,2)),np.array([[1]]))).reshape((3,1))
            pr=np.linalg.inv(H2).dot(rpr)[0:2]/np.linalg.inv(H2).dot(rpr)[2]
            left.append(pl[0:2].reshape((1,2)))
            right.append(pr[0:2].reshape((1,2)))
        left_points.append(left)
        right_points.append(right)

    if verbose:
        print(f"Original points remapped")  

    points=[]
    for  uvs_l,uvs_r in zip(left_points,right_points):
        points.append(triangulate_points(mtxl,mtxr,R,T,[uvs_l],[uvs_r]))



    if verbose:
        print('GOT 3D POINTS')

        img1=cv2.imread(left_image_path)
        img2=cv2.imread(right_image_path)
        for pl,pr in zip(left_points[0],right_points[0]):
            centerl=[int(pl[0][0]),int(pl[0][1])]
            centerr=[int(pr[0][0]),int(pr[0][1])]
            cv2.circle(img1, centerl, 5, (0, 255, 0), -1)
            cv2.circle(img2, centerr, 5, (0, 255, 0), -1)
        img=np.hstack((img1,img2))
        cv2.imwrite('test_'+str(left_image_path[-10:-5])+'_.jpg',img)

    vecs=[]
    for person in points:
        vec=[]
        # vec_inds=[[5,6],[5,7],[5,11],[6,8],[6,12],[7,9],[8,10],[11,12],[11,13],[12,14],[13,15],[14,16]]
        vec_inds=[[6,5],[6,8],[8,10],[5,7],[7,9],[12,14],[14,16],[11,13],[13,15]]
        for pair in vec_inds:
            vec.append(np.linalg.norm(person[pair[0]].reshape((1,3))[0]-person[pair[1]].reshape((1,3))[0]))
        vecs.append(vec)
    if verbose:
        print('GOT VECS')
        print(vecs)

    return vecs





# Pull required data from settings file (for now, added manually)

# Left intrinsic matrix
## Can be optimized with box/laser/caliper or simply use matrix from initial single cam calibration
mtxl=np.array([[1364.45428, 0.0, 1037.92385],
               [0.0, 1354.46497, 394.715706],
               [0.0,0.0,1.0]],dtype='float32')


## Right intrinsic matrix
### Can be optimized with box/laser/caliper or simply use matrix from initial single cam calibration
mtxr=np.array([[1402.122679830661,0.0,981.7301190980668],
               [0.0,1392.888966296978,493.14964335781156],
               [0.0,0.0,1.0]],dtype='float32')


## Rotation extrinsic matrix
R=np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])

## Translation extrinsic matrix
T=np.array([[65.0],[0.0],[0.0]])

## Left rectification homography
### Can be computed with VIPER-REGISTRAR/lib/camera_calibration/rectification_point_method.py


H1=np.array(
 [[-8.75462841e-03,  2.51283566e-04,  1.92743788e+00],
 [ 4.81464217e-04, -9.99169278e-03, -1.64847260e+00],
 [ 2.52817541e-07,  2.04679142e-07, -1.06762142e-02]])


H2=np.array(
  [[ 9.56397691e-01,  3.91829334e-02,  2.06994327e+01],
 [-6.49897681e-02,  9.98176305e-01,  6.33749728e+01],
 [-4.45459612e-05, -1.82501636e-06,  1.04374963e+00]]
)




left_image_path='./3969.jpeg'
right_image_path='./3969_stereo.jpeg'

vec=person_vecs_identification(mtxl,mtxr,R,T,H1,H2,left_image_path,right_image_path,verbose=True)

print(f'vecs:{vec}')




