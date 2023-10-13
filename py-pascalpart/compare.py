import os

image_folder = "./datasets/images/train2017"
label_folder = "./datasets/labels/train2017"

# Get a list of image files in the image folder
image_files = [file for file in os.listdir(image_folder) if file.endswith(".png")]

# Get a list of label files in the label folder
label_files = [file for file in os.listdir(label_folder) if file.endswith(".txt")]

# Remove the file extensions to compare
image_names = [os.path.splitext(file)[0] for file in image_files]
label_names = [os.path.splitext(file)[0] for file in label_files]

# Check if each image has a corresponding label
missing_labels = [image_name for image_name in image_names if image_name not in label_names]

if not missing_labels:
    print("All images have corresponding labels.")
else:
    print("Missing labels for the following images:")
    for missing_label in missing_labels:
        print(missing_label)