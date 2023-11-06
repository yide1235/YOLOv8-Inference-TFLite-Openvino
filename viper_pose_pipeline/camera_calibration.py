import glob
# import corner_detect
# import detect_board_situation
# import get_mask_cutoff
import numpy as np
import cv2
import time

def camera_calibration(frames_path,square_size,board_size):
    paths=glob.glob(frames_path+'*')
    img_points=[]
    obj_points=[]
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size   
    mean_error=0
    paths_found=[]
    for path in paths:
        print("finding corner of ", path)
        # Get the corners of the checkerboard in the image
        gray = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        if ret:
            paths_found.append(path)
            img_points.append(np.array(corners,np.float32))
            obj_points.append(objp)
    if len(img_points)==0:
        return [],[]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None, flags=cv2.CALIB_USE_LU)
    
    errors=[]
    for i in range(len(obj_points)):
        
        mean_error=0
        img_points_proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(img_points[i], img_points_proj, cv2.NORM_L2) / len(img_points_proj)
        mean_error += error
        mean_reprojection_error = mean_error / len(obj_points)
        errors.append(mean_reprojection_error)
    errors=np.array(errors)
    mean_errors=np.mean(errors)
    q1 = np.percentile(errors, 25)
    q3 = np.percentile(errors, 75)
    iqr=q3-q1
    upper_limit=1.5*iqr+q3
    good_inds=np.where(errors < upper_limit)[0]

    good_paths=[value for value in paths_found if paths_found.index(value) in good_inds]

    img_points=[]
    obj_points=[]

    for path in good_paths:
        # Get the corners of the checkerboard in the image
        print("calibrating on ", path)
        gray = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        if ret:
            img_points.append(np.array(corners,np.float32))
            obj_points.append(objp)

    if len(img_points)==0:
        return [],[]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None, flags=cv2.CALIB_USE_LU)

    return np.array(mtx),dist




def camera_calibration_2(frames_path,ext=".jpg",small_square_size=5,big_square_size=20, detect_tol=5,detect_margin=20,detect_crop_size=24,detect_crop_step=1,detect_crop_threshold=0.1, detect_crop_var_tol=2500,detect_crop_window_p=0.05,detect_crop_raw_size=10,detect_crop_raw_stride=1, detect_crop_raw_threshold=0.07,corner_crop_window_percentage=0.05, corner_raw_box_size=10,corner_raw_step=1,corner_raw_threshold=0.07,corner_score_box_size=4,corner_score_variance_tol=30, corner_score_average=True,corner_score_grouping_distance=6,refine_corner_box_size=20):
    """
    Takes in:
        - frames_path: path to frames (ex: folder path, images/calibration_images/)
        - ext: file extention for images (ex: .jpg)
        - small_square_size: dimension in desired units (often mm) for smaller checkerboard squares
        - large_square_size: similar to above, however for large squares (if all images have same size squares set both values the same)
        several other variables are used, please refer to the corresponding functions where the value is used for descriptions
    Returns:
        - mtx: intrinsic camera matrix
        - dist: distortion coefficients for camera
    """
    img_points=[]
    obj_points=[]
    rows=[]
    cols=[]
    paths=glob.glob(frames_path+"*")
    got=[0]*len(paths) 
    for path in paths:
        frame=path.split(ext)[0]
        try:
            start=time.time()
            mask_val=get_mask_cutoff.get_mask_cutoff(path)
            end=time.time()
            print(f'GETTING MASK VALUE TOOK {end-start} seconds')
            start=time.time()
            row,col=detect_board_situation.get_board_dimensions(frame,ext,tol=detect_tol,margin=detect_margin,crop_size=detect_crop_size,crop_step=detect_crop_step,crop_threshold=detect_crop_threshold,crop_var_tol=detect_crop_var_tol,crop_window_p=detect_crop_window_p,raw_size=detect_crop_raw_size,raw_stride=detect_crop_raw_stride,raw_threshold=detect_crop_raw_threshold,mask_cutoff=mask_val)
            end=time.time()
            print(f'DETECTING BOARD SITUATION TOOL {end-start} seconds')
        except Exception as error:
            print("error at board detection")
            print(error)
            continue
#         print("Board dim: ",row," x ",col)

        if row!=0:
            try:
                # Get the corners of the checkerboard in the image
                gray = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (row,col), None)
                if ret:
                    img_points.append(np.array(corners,np.float32))
                    rows.append(row)
                    cols.append(col)
                    print(f'FOUND:{frame}')
                    got[i]=1
                    break
                else:
                    print(f'NOT FOUND:{frame}')
                    got[i]=0
            except Exception as error:
                print("error at corner detection")
                print(error)
        # else:
        #     print("ROW COUNT WAS 0")

    if len(img_points)==0:
        return [], []
    
    dist=[]
    # Find the mean size of squares in all images in pixel distance
    for i in range(len(img_points)):
        dist.append(np.linalg.norm([img_points[i][0][0][0]-img_points[i][1][0][0],img_points[i][0][0][1]-img_points[i][1][0][1]]))
    mid_size=np.mean(dist)
    
    # Identify if the squares in image are large (greater than mean) or small (less than mean) and assign corresponding world scale
    for i in range(len(img_points)):
        if dist[i]<=mid_size:
            square_size=small_square_size
        else:
            square_size=big_square_size
            
        objp = np.zeros((rows[i] * cols[i], 3), np.float32)
        objp[:, :2] = np.mgrid[0:rows[i], 0:cols[i]].T.reshape(-1, 2) * square_size   
        obj_points.append(objp)

    print(f'Starting calibration with {len(img_points)} successful boards detected')
    start=time.time()
    # Get intrinsic matrix and distortion coefficients using image points and corresponding world points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None, flags=cv2.CALIB_USE_LU)
    end=time.time()
    print(f'Calibration took {end-start} seconds')
    
    return np.array(mtx),dist,got,img_points



# import json
# import glob
# import random
# config_file_path='config_files/config.json'
# with open(config_file_path,'r') as f:
#     params=json.load(f)
# for param in params:
#     globals()[param]=params[param]
# file_calib_ind_order=list(range(len(glob.glob('tmp_images/'+'*'))))
# random.shuffle(file_calib_ind_order)
# # Get intrinsic matrices and distortion coefficients for both the left and right camera
# mtxl,distl=camera_calibration(file_calib_ind_order,max_single_cam_good_images,frames_path='tmp_images/',ext=ext, small_square_size=small_square_size, big_square_size=big_square_size, detect_tol=detect_tol,detect_margin=detect_margin,detect_crop_size=detect_crop_size, detect_crop_step=detect_crop_step,detect_crop_threshold=detect_crop_threshold, detect_crop_var_tol=detect_crop_var_tol,detect_crop_window_p=detect_crop_window_p, detect_crop_raw_size=detect_crop_raw_size,detect_crop_raw_stride=detect_crop_raw_stride, detect_crop_raw_threshold=detect_crop_raw_threshold,  corner_crop_window_percentage=corner_crop_window_percentage,corner_raw_box_size=corner_raw_box_size, corner_raw_step=corner_raw_step,corner_raw_threshold=corner_raw_threshold, corner_score_box_size=corner_score_box_size,corner_score_variance_tol=corner_score_variance_tol, corner_score_average=corner_score_average,corner_score_grouping_distance=corner_score_grouping_distance)
    
