import numpy as np
from scipy.ndimage import uniform_filter
import time

def pad_image(image, padding):
    padded_height = image.shape[0] + 2 * padding
    padded_width = image.shape[1] + 2 * padding
    padded_image = np.zeros((padded_height, padded_width), dtype=image.dtype)
    padded_image[padding:-padding, padding:-padding] = image
    return padded_image

def smoother(image, filter_size):
    height, width = image.shape
    half_filter_size = filter_size // 2

    # Create a padded image to simplify boundary handling
    padded_image = pad_image(image, half_filter_size)

    # Create a filter kernel
    kernel = np.ones((filter_size, filter_size)) / (filter_size ** 2)

    # Convolve the padded image with the kernel
    blurred_image = np.zeros_like(image, dtype=float)

    for y in range(height):
        for x in range(width):
            y_start = y
            y_end = y + filter_size
            x_start = x
            x_end = x + filter_size

            neighborhood = padded_image[y_start:y_end, x_start:x_end]
            blurred_value = np.sum(neighborhood * kernel)

            blurred_image[y, x] = blurred_value

    return blurred_image




def ssim(im1,im2,channel_axis,multichannel,win_size):
    # im1=im1.astype('float32')
    # im2=im2.astype('float32')
    # im1=im1.astype('float32')/255.0
    # im2=im2.astype('float32')/255.0


    # print('-----',im1.shape, im2.shape)
    if multichannel==True:
        mssims=[]
        for i in range(im1.shape[channel_axis]):
            mssims.append(ssim(im1[:,:,i],im2[:,:,i],None,False,win_size))
        # print(mssims)
        return np.mean(mssims)
    else:
        ux=smoother(im1,win_size)
        uy=smoother(im2,win_size)
        uxx=smoother(im1*im1,win_size)
        uxy=smoother(im1*im2,win_size)
        uyy=smoother(im2*im2,win_size)
        NP=win_size**2
        cov_norm=NP/(NP-1)
        vx=cov_norm*(uxx-ux*ux)
        vy=cov_norm*(uyy-uy*uy)
        vxy=cov_norm*(uxy-ux*uy)
        k1=0.01
        k2=0.03
        R=2
        c1=(k1*R)**2
        # print(c1)
        c2=(k2*R)**2
        A1=2*ux*uy+c1
        # print(im2[10,10])
        A2=2*vxy+c2
        B1=ux*ux+uy*uy+c1
        B2=vx+vy+c2
        ssim_mat=A1*A2/(B1*B2)
        pad=(win_size-1)//2
        mssim=np.mean(ssim_mat[pad:im1.shape[0]-pad,pad:im1.shape[1]-pad])
        return mssim
    




# import cv2

# imgl=cv2.imread('../../Dropbox/REALTIME7/tmpl.jpg').astype('float32')
# imgr=cv2.imread('../../Dropbox/REALTIME7/tmpr.jpg').astype('float32')


# image1 = imgl
# image2 = imgr

# p1=[100,1000]
# size=10
# target_col=1090
# im1=image1[p1[1]-size:p1[1]+size+1,p1[0]-size:p1[0]+size+1]
# ssims=[]
# im2=image2[p1[1]-size:p1[1]+size+1,1040-size:1040+size+1]

# print("here")
# print(ssim(im1,im2,2,True,7))


