import argparse
import PIL
import PIL.Image
import shutil
import os
# import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import cv2


import numpy as np
import imantics
from imantics import Polygons, Mask


SHOW_IMAGES = True

import scipy.io



#Copy code
parcal_part_class_names = [
    'hair',
    'head',
    'lear',
    'leye',
    'lebrow',
    'lfoot',
    'lhand',
    'llarm',
    'llleg',
    'luarm',
    'luleg',
    'mouth',
    'neck',
    'nose',
    'rear',
    'reye',
    'rebrow',
    'rfoot',
    'rhand',
    'rlarm',
    'rlleg',
    'ruarm',
    'ruleg',
    'torso'
]


# Load annotations from .mat files creating a Python dictionary:
def load_annotations(path):

    # Get annotations from the file and relative objects:
    annotations = scipy.io.loadmat(path)["anno"]

    objects = annotations[0, 0]["objects"]

    # List containing information of each object (to add to dictionary):
    objects_list = []

    # Go through the objects and extract info:
    for obj_idx in range(objects.shape[1]):
        obj = objects[0, obj_idx]

        # Get classname and mask of the current object:
        classname = obj["class"][0]
        mask = obj["mask"]

        # List containing information of each body part (to add to dictionary):
        parts_list = []

        parts = obj["parts"]

        # Go through the part of the specific object and extract info:
        for part_idx in range(parts.shape[1]):
            part = parts[0, part_idx]
            # Get part name and mask of the current body part:
            part_name = part["part_name"][0]
            part_mask = part["mask"]

            # Add info to parts_list:
            parts_list.append({"part_name": part_name, "mask": part_mask})

        # Add info to objects_list:
        objects_list.append({"class": classname, "mask": mask, "parts": parts_list})

    return {"objects": objects_list}




# Load annotations from the annotation folder of PASCAL-Part dataset:
if __name__ == "__main__":
    # Parse arguments from command line:
    parser = argparse.ArgumentParser(description="Extract data from PASCAL-Part Dataset")
    # parser.add_argument("--annotation_folder", default="./datasets/Annotations_Part", help="Path to the PASCAL-Part Dataset annotation folder")
    # parser.add_argument("--images_folder", default="./datasets/VOCdevkit/VOC2010/JPEGImages", help="Path to the PASCAL VOC 2010 JPEG images")
    
    # parser.add_argument("--annotation_folder", default="./mask", help="Path to the PASCAL-Part Dataset annotation folder")
    # parser.add_argument("--images_folder", default="./image", help="Path to the PASCAL VOC 2010 JPEG images")
    parser.add_argument("--annotation_folder", default="./person_mask", help="Path to the PASCAL-Part Dataset annotation folder")
    parser.add_argument("--images_folder", default="./person_image", help="Path to the PASCAL VOC 2010 JPEG images")

    args = parser.parse_args()

    # Stats on the dataset:
    obj_cnt = 0
    # bodypart_cnt = 0

    mat_filenames = os.listdir(args.annotation_folder)
    # img_filenames = os.listdir(args.images_folder)
    # Iterate through the .mat files contained in path:

    for idx, annotation_filename in enumerate(mat_filenames):
        annotations =load_annotations(os.path.join(args.annotation_folder, annotation_filename))
        img_name=annotation_filename[:annotation_filename.rfind(".")]
        image_filename = annotation_filename[:annotation_filename.rfind(".")] + ".jpg" # PASCAL VOC image have .jpg format


        obj_cnt =0

        # Show original image with its mask:
        img = PIL.Image.open(os.path.join(args.images_folder, image_filename))
        # print(annotations["objects"])
        image_width, image_height = img.size

        # print(image_width, image_height)

        # image_filename=image_filename.replace(".jpg","")
        with open('./datasets/labels/train2017/{}.txt'.format(img_name), 'a') as file:


            bodypart_cnt = 0
            for obj in annotations["objects"]:
                

                #this is only for the person class            
                # # Check if the class is "person"
                # if any(obj["class"] == "person" for obj in annotations["objects"]):
                #     # Copy the .mat file to the person_mask folder
                #     shutil.copy(os.path.join(args.annotation_folder, annotation_filename), "./person_mask")
                #     shutil.copy(os.path.join(args.images_folder, image_filename), "./person_image")
                
                if obj["class"] == "person":     
                    # print(obj["class"]) 
                    obj_cnt+= len(annotations["objects"])
                    bodypart_cnt += len(obj["parts"])
                    print("obj_cnt: {} - bodypart_cnt: {}".format(obj_cnt, bodypart_cnt), end="\r")

                    # if SHOW_IMAGES:

                    for body_part in obj["parts"]:

                        #img, mask, bodypart_mask, windowtitle, suptitle
                        #plot_mask(img, obj["mask"], body_part["mask"], image_filename, obj["class"] + ": " + body_part["part_name"])
                        mask=obj["mask"]
                        bodypart_mask=body_part["mask"]

                        mask_array = np.array(mask) * 255  # Convert mask to NumPy array and scale to 0-255 range
                        bodypart_mask_array = np.array(bodypart_mask) * 255  # Convert bodypart_mask to NumPy array and scale to 0-255 range

                        # Save the images using cv2.imwrite
                        cv2.imwrite(f"./datasets/images/train2017/{image_filename}", np.array(img))  # Save img as a JPEG image

                        # cv2.imwrite(f"./{image_filename}_mask.jpg", mask_array)  # Save mask as a grayscale image
                        part_name=body_part["part_name"]
                        # cv2.imwrite(f"./{image_filename}_{obj_cnt}_{part_name}_bodyparts.jpg", bodypart_mask_array)  # Save bodypart_mask as a grayscale image
                        class_id=parcal_part_class_names.index(body_part["part_name"])

                        
                        polygons = Mask(bodypart_mask_array).polygons()
                        # print(polygons.points)
                        # print(polygons.segmentation)
                        points_list=polygons.segmentation

                            # Normalize the coordinates and format with three decimal places
                        normalized_points = [("{:.3f}".format(x / image_width), "{:.3f}".format(y / image_height)) for x, y in zip(points_list[0][::2], points_list[0][1::2])]

                        # Flatten the list of formatted normalized coordinates
                        normalized_points_flat = [float(coord) for point in normalized_points for coord in point]

                        # Print the normalized points as a single list of floats
                        normalized_points_flat = str(normalized_points_flat).replace("[", "").replace("]", "").replace(",", "")
                        if((class_id!=None)and (normalized_points_flat!=None)):
                            file.write(str(class_id)+" ")
                            file.write(normalized_points_flat+"\n")
     

                        # plot_mask(img, obj["mask"], body_part["mask"], image_filename, obj["class"] + ": " + body_part["part_name"])


    print("obj_cnt: {} - bodypart_cnt: {}".format(obj_cnt, bodypart_cnt))
