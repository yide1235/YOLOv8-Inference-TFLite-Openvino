

import numpy as np
import cv2
# from skimage.metrics import structural_similarity as ssim
import sys
sys.path.append('./LINE_MATCHING/')
from my_ssim import ssim
from SSIM_PIL import compare_ssim
from PIL import Image

# def sweep_line_block(src_img, trg_img, src_point, trg_point, height_radius, width_radius, shift):
#     row = round(src_point[1])
#     src_col = round(src_point[0])
#     src_block = src_img[row-height_radius:row+height_radius+1, src_col-width_radius:src_col+width_radius+1]

#     trg_start = max(round(trg_point[0]) - shift, 0)
#     trg_end = min(round(trg_point[0]) + shift, trg_img.shape[1])

#     max_one = trg_start
#     max_ssim = 0

#     max_height=src_point[1]

#     i = trg_start
    
#     while i < trg_end:

#         row = round(src_point[1])
        
#         for row in range(row-1, row+2):

#             row_start = max(row - height_radius, 0)
#             row_end = min(row + height_radius + 1, trg_img.shape[0])
#             col_start = max(i - width_radius, 0)
#             col_end = min(i + width_radius + 1, trg_img.shape[1])

#             # Calculate padding if necessary
#             pad_top = max(0, height_radius - row)

#             # Extract and pad trg_block
#             trg_block = trg_img[row_start:row_end, col_start:col_end]


#             src_block = (src_block * 255).astype('uint8')
#             trg_block = (trg_block * 255).astype('uint8')

#             src_block_image = Image.fromarray(src_block)
#             trg_block_image = Image.fromarray(trg_block)

#             score_ssim=compare_ssim(src_block_image , trg_block_image, GPU=True)

            

#             if score_ssim > max_ssim:
#                 max_one = i
#                 max_ssim = score_ssim
#                 max_height=row
#             print(row,i,score_ssim,max_one, max_height)
            
#             i += 1
#     print('-----------------')

#     return [max_one, max_height]






def sweep_line_block(src_img, trg_img, src_point, trg_point, height_radius, width_radius, shift):
    row = round(src_point[1])
    src_col = round(src_point[0])
    scores = []
    # Padding for src_block
    src_row_start = max(0, row - height_radius)
    src_row_end = min(src_img.shape[0], row + height_radius + 1)
    src_col_start = max(0, src_col - width_radius)
    src_col_end = min(src_img.shape[1], src_col + width_radius + 1)
    # print(src_row_start, src_row_end,src_col_start,src_col_end )
    src_block = src_img[src_row_start:src_row_end, src_col_start:src_col_end]

    pad_top = max(0, height_radius - row)
    pad_bottom = max(0, row + height_radius + 1 - src_img.shape[0])
    pad_left = max(0, width_radius - src_col)
    pad_right = max(0, src_col + width_radius + 1 - src_img.shape[1])

    src_block = cv2.copyMakeBorder(src_block, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

    trg_start = max(round(trg_point[0]) - shift, 0)
    trg_end = min(round(trg_point[0]) + shift, trg_img.shape[1])

    max_one = trg_start
    max_ssim = 0
    max_height=src_point[1]

    i = trg_start
    while i < trg_end:

        for row in range(row-5, row+6):
            # Padding for trg_block
            trg_row_start = max(0, row - height_radius)
            trg_row_end = min(trg_img.shape[0], row + height_radius + 1)
            trg_col_start = max(0, i - width_radius)
            trg_col_end = min(trg_img.shape[1], i + width_radius + 1)

            trg_block = trg_img[trg_row_start:trg_row_end, trg_col_start:trg_col_end]

            pad_top = max(0, height_radius - row)
            pad_bottom = max(0, row + height_radius + 1 - trg_img.shape[0])
            pad_left = max(0, width_radius - i)
            pad_right = max(0, i + width_radius + 1 - trg_img.shape[1])

            trg_block = cv2.copyMakeBorder(trg_block, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)


            # Compute SSIM
            # score_ssim = ssim(src_block, trg_block, multichannel=True, channel_axis=2, win_size=7)
            
            
            #use python ssim library
            src_block = (src_block * 255).astype('uint8')
            trg_block = (trg_block * 255).astype('uint8')

            # print(trg_block)
            # print(type(trg_block))
            # print(trg_block.shape)

            src_block_image = Image.fromarray(src_block)

            trg_block_image = Image.fromarray(trg_block)


            #the weights method
            # weighted_src_block_image = apply_center_weighted_filter(src_block_image)
            # weighted_trg_block_image = apply_center_weighted_filter(trg_block_image)
                        

            score_ssim=compare_ssim(src_block_image , trg_block_image, GPU=True)

            # #the weights method
            # score_ssim=compare_ssim(weighted_src_block_image , weighted_trg_block_image, GPU=True)


            scores.append(score_ssim)

            if score_ssim > max_ssim:
                max_one = i
                max_ssim = score_ssim
                max_height=row
            i += 1

    return [max_one, max_height]



# def weight_filter








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


