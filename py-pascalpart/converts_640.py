from PIL import Image
import os

def resize_images(source_folder, target_folder, size=(640, 640)):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for filename in os.listdir(source_folder):
        if filename.endswith('.png'):
            img_path = os.path.join(source_folder, filename)
            img = Image.open(img_path)
            img_resized = img.resize(size, Image.LANCZOS)

            save_path = os.path.join(target_folder, filename)
            img_resized.save(save_path)

if __name__ == "__main__":
    source_folder = './test_cropped'
    target_folder = './cropped_640'

    resize_images(source_folder, target_folder)
