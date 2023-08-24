from keras_cv_attention_models import yolov8
import tensorflow as tf


model = yolov8.YOLOV8_L(pretrained="coco")




# # Run prediction
from keras_cv_attention_models import test_images
imm = test_images.dog_cat()
print(imm.shape)
#input of the model is (1,640,640,3)
temp=model.preprocess_input(imm)
print(temp.shape)

preds = model(temp)
bboxs, labels, confidences = model.decode_predictions(preds)[0]
print(bboxs.shape)
print(labels)
print(confidences.shape)


# # Show result
# from keras_cv_attention_models.coco import data
# data.show_image_with_bboxes(imm, bboxs, lables, confidences)

# def representative_data_gen():
#   for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
#     yield [input_value]


##get the tflite file
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations=[tf.lite.Optimize.DEFAULT]
# tflite_model = converter.convert()
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.representative_dataset = representative_data_gen
# # Ensure that if any ops can't be quantized, the converter throws an error
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# # Set the input and output tensors to uint8 (APIs added in r2.3)
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8

# tflite_model_quant = converter.convert()

# interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
# input_type = interpreter.get_input_details()[0]['dtype']
# print('input: ', input_type)
# output_type = interpreter.get_output_details()[0]['dtype']
# print('output: ', output_type)

# import pathlib

# tflite_models_dir = pathlib.Path("/home/myd/Desktop/")
# tflite_models_dir.mkdir(exist_ok=True, parents=True)

# # Save the unquantized/float model:
# # tflite_model_file = tflite_models_dir/"mnist_model.tflite"
# # tflite_model_file.write_bytes(tflite_model)
# # Save the quantized model:
# tflite_model_quant_file = tflite_models_dir/"yolov8_model_quant.tflite"
# tflite_model_quant_file.write_bytes(tflite_model_quant)





# tflite_model_path = './yolov8_l_quant.tflite'
# with open(tflite_model_path, 'wb') as f:
#     f.write(tflite_model_quant)