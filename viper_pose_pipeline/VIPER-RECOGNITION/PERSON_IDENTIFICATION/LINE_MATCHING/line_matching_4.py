

import numpy as np
import cv2
# from skimage.metrics import structural_similarity as ssim
import sys
sys.path.append('./LINE_MATCHING/')
from my_ssim import ssim




def sweep_line_block(src_img,trg_img,bboxes_r, bboxes_l,src_point,height_radius,width_radius):
    row=int(src_point[1])
    src_col=int(src_point[0])
    src_block=src_img[row-height_radius:row+height_radius+1,src_col-width_radius:src_col+width_radius+1]
    scores=[]
    # for i in range(width_radius,trg_img.shape[1]-width_radius):
    for i in range(round(bboxes_l[0][0])+width_radius,round(bboxes_l[0][2])-width_radius):
        trg_block=trg_img[row-height_radius:row+height_radius+1,i-width_radius:i+width_radius+1]
        # print(trg_block==0.0)
        
        score_ssim = ssim(src_block, trg_block, multichannel=True, channel_axis=2, win_size=7)
        scores.append(score_ssim)
        print(i, score_ssim)

    return [scores.index(max(scores))+width_radius,src_point[1]]




# # src_image=cv2.imread('../../Dropbox/REALTIME7/tmpl.jpg')
# # trg_image=cv2.imread('../../Dropbox/REALTIME7/tmpr.jpg')
# # src_point=[1148,742]
# # height_radius=10
# # width_radius=10

# # print(sweep_line_block(src_image,trg_image,src_point,height_radius,width_radius))



# def sweep_line_block(src_img, trg_img, src_point, height_radius, width_radius):
#     row = int(src_point[1])
#     src_col = int(src_point[0])
#     src_block = src_img[row - height_radius:row + height_radius + 1, src_col - width_radius:src_col + width_radius + 1]
#     scores = []
#     for i in range(width_radius, trg_img.shape[1] - width_radius):
#         trg_block = trg_img[row - height_radius:row + height_radius + 1, i - width_radius:i + width_radius + 1]
        
#         # Validate the image blocks
#         if src_block.size == 0 or trg_block.size == 0:
#             print(f"Invalid block at index {i}: src_block or trg_block is empty.")
#             continue

#         score_ssim = ssim(src_block, trg_block, multichannel=True, channel_axis=2, win_size=7)
        
#         if score_ssim is not None:
#             scores.append(score_ssim)
#         else:
#             print(f"Warning: SSIM returned None for index {i}.")
#             scores.append(-1)  # Append a default low score or handle it as needed

#     if scores:
#         best_index = scores.index(max(scores)) + width_radius
#     else:
#         print("No valid SSIM scores were computed.")
#         best_index = -1  # Handle the case when no scores are computed
    
#     return [best_index, src_point[1]]





    