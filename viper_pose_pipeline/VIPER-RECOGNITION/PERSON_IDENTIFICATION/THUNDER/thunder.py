


# Import TF and TF Hub libraries.
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image

# Load the input image.
# image_path = 'tmp.jpg'
def run_thunder(image_path):
    image = tf.io.read_file(image_path)
    image = tf.compat.v1.image.decode_jpeg(image)
    image = tf.expand_dims(image, axis=0)

    dim_prior=image.shape
    if dim_prior[1]>dim_prior[2]:
        y_shift=0
        x_shift=(256-(dim_prior[2]/(dim_prior[1]/256)))//2
        scale=dim_prior[1]/256
    else:
        y_shift=(256-(dim_prior[1]/(dim_prior[2]/256)))//2
        x_shift=0
        scale=dim_prior[2]/256

    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    image = tf.cast(tf.image.resize_with_pad(image, 256, 256), dtype=tf.uint8)

    # Save the image using a suitable library (PIL, OpenCV, etc.)
    # Here, we'll use PIL (Python Imaging Library):
    output_image_path = 'model_input.jpg'
    image_array = tf.squeeze(image, axis=0).numpy().astype(np.uint8)
    result_image = Image.fromarray(image_array)
    # result_image.save(output_image_path)


    # # Download the model from TF Hub.
    # model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")

    # movenet = model.signatures['serving_default']
    # # Run model inference.
    # outputs = movenet(image)

    model_path = "./THUNDER/model.tflite"
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    keypoints = interpreter.get_tensor(output_details[0]['index'])



    # Compute keypoints location in image
    # keypoints = outputs['output_0']
    points=[]
    for point in keypoints[0][0]:
        # print(point)
        points.append([int(-x_shift+(point[1]*256))*scale,int(-y_shift+(point[0]*256))*scale])
    return [points]
















