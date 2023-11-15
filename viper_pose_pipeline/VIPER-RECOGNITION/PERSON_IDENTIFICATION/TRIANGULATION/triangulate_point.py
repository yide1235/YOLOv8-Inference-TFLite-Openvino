import numpy as np
def triangulate_point(M_r,P_l,point_l,point_r):
    A_b=np.asmatrix(np.zeros((4,3)))
    A_b[0,0]=point_r[0]*M_r[2,0]-M_r[0,0]
    A_b[0,1]=point_r[0]*M_r[2,1]-M_r[0,1]
    A_b[0,2]=point_r[0]*M_r[2,2]-M_r[0,2]
    A_b[1,0]=point_r[1]*M_r[2,0]-M_r[1,0]
    A_b[1,1]=point_r[1]*M_r[2,1]-M_r[1,1]
    A_b[1,2]=point_r[1]*M_r[2,2]-M_r[1,2]
    A_b[2,0]=point_l[0]*P_l[2,0]-P_l[0,0]
    A_b[2,1]=point_l[0]*P_l[2,1]-P_l[0,1]
    A_b[2,2]=point_l[0]*P_l[2,2]-P_l[0,2]
    A_b[3,0]=point_l[1]*P_l[2,0]-P_l[1,0]
    A_b[3,1]=point_l[1]*P_l[2,1]-P_l[1,1]
    A_b[3,2]=point_l[1]*P_l[2,2]-P_l[1,2]
    A_b=np.array(A_b)

    b=np.asmatrix(np.zeros((4,1)))
    b[0,0]=M_r[0,3]-M_r[2,3]
    b[1,0]=M_r[1,3]-M_r[2,3]
    b[2,0]=P_l[0,3]-P_l[2,3]
    b[3,0]=P_l[1,3]-P_l[2,3]
    b=np.array(b)
    try:
        X=np.linalg.inv(np.transpose(A_b).dot(A_b)).dot(np.transpose(A_b)).dot(b)
    except:
        X=np.linalg.pinv(np.transpose(A_b).dot(A_b)).dot(np.transpose(A_b)).dot(b)
    return X

def triangulate_points(K_l,K_r,R,T,uvs_l,uvs_r):
    M_l=np.hstack([K_l, np.array([[0], [0], [0]])])
    M_r=np.hstack([K_r, np.array([[0], [0], [0]])])
    RT=np.vstack([np.hstack([R,T]),[0,0,0,1]])
    P_l=M_l.dot(RT)
    # print(M_l,RT,P_l)
    
    p3d_cam=[]
    for i in range(len(uvs_l)):
        for j in range(len(uvs_l[0])):
            # print(uvs_l[i][j][0],uvs_r[i][j][0])
            point=triangulate_point(M_r,P_l,uvs_l[i][j][0],uvs_r[i][j][0])
            p3d_cam.append(point)
        
    return p3d_cam




# mtxl=np.array([[1381.35935,0,941.40662],
#                [0,1381.35935*1.01825,470.19631],
#                [0,0,1]],dtype='float32')

# ## Right intrinsic matrix
# ### Can be optimized with box/laser/caliper or simply use matrix from initial single cam calibration
# mtxr=np.array([[1377.64593,0,889.06106],
#                [0,1377.64593*1.01796,499.96914],
#                [0,0,1]],dtype='float32')

# ## Rotation extrinsic matrix
# R=np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])

# ## Translation extrinsic matrix
# T=np.array([[65.0],[0.0],[0.0]])

# H1=np.array(
#  [[ 4.22772245e-02, -1.30516921e-02, -8.08973235e+00],
#  [ 1.94229434e-02,  2.14953874e-02, -1.53770719e+01],
#  [ 1.83075763e-05, -8.81994917e-06 , 1.42816432e-02]])
# H2=np.array(
#  [[ 1.59147194e+00, -5.66628608e-01 ,-2.61833611e+02],
#  [ 7.00703672e-01 , 8.12012879e-01 ,-5.71162480e+02],
#  [ 6.76459965e-04, -2.40847205e-04  ,4.80655924e-01]])


# p1=np.linalg.inv(H1).dot(np.array([[1000],[1000],[1]]))
# p2=np.linalg.inv(H2).dot(np.array([[837],[1000],[1]]))

# print("here")
# p1=[p1[0]/p1[2],p1[1]/p1[2]]
# p2=[p2[0]/p2[2],p2[1]/p2[2]]
# print(p1,p2)

# print("here")


# print(triangulate_points(mtxl,mtxr,R,T,[[[p1]]],[[[p2]]]))