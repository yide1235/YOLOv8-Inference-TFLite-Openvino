import math
import time
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import os
import glob

from YOLOSeg_bg import YOLOSeg_bg
from YOLOSeg_humanpart import YOLOSeg_humanpart 


def bg_removal():
    count=0

    yoloseg = YOLOSeg_bg()

    # input_folder='./test_fullimage2'
    input_folder='./test'

    image_files = glob.glob(f'{input_folder}/*.[jp][pn][ge]')

    for i in image_files:

        frame=cv2.imread(i)

        if frame is None:
            print(f"Error: Unable to read the image {i}")
            continue  # Move to the next iteration of the loop if the image cannot be read



        boxes, scores, class_ids, masks = yoloseg(frame)

        # postprocess and draw masks
        combined_img = yoloseg.draw_output(frame)

        cv2.imwrite("./out/"+str(count)+'.png',combined_img)


        count+=1
        

    # Press Any key stop
    if cv2.waitKey(1) > -1:
        
        print("finished by user")

    cv2.destroyAllWindows()




def humanpart():
    count=0

    yoloseg = YOLOSeg_humanpart()

    # input_folder='./test_fullimage2'
    input_folder='./out'

    image_files = glob.glob(f'{input_folder}/*.[jp][pn][ge]')

    for i in image_files:

        frame=cv2.imread(i)

        if frame is None:
            print(f"Error: Unable to read the image {i}")
            continue  # Move to the next iteration of the loop if the image cannot be read



        boxes, scores, class_ids, masks = yoloseg(frame)

        # print('---------------')
        # print(boxes)
        # print(boxes.shape)
        # print('---------------')
        # print(scores)
        # print('---------------')
        # print(class_ids)
        # print('---------------')
        # print(masks)
        # print(type(masks))
        # print(masks.shape)
        # print('---------------')


        # postprocess and draw masks
        combined_img = yoloseg.draw_output(frame)

        cv2.imwrite("./second_out/"+str(count)+'.png',combined_img)


        count+=1
        

    # Press Any key stop
    if cv2.waitKey(1) > -1:
        
        print("finished by user")

    cv2.destroyAllWindows()


def main():
    bg_removal()
    humanpart()


if __name__ == '__main__':
    main()
