from PIL import Image
import os

def resize_images(source_folder, target_folder):
    # Ensure target folder exists or create it
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for filename in os.listdir(source_folder):
        # Assuming we're only interested in images, and to simplify the example, 
        # let's process files that end with common image extensions
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(source_folder, filename)
            
            # Open the image
            img = Image.open(img_path)
            
            # Get original dimensions
            original_size = img.size
            
            # Downsample to 640x640
            # img_downsampled = img.resize((360, 360), Image.LANCZOS)
            img_downsampled = img.resize((300,300))
            # Upsample to its original size
            img_upsampled = img_downsampled.resize(original_size, Image.LANCZOS)
            
            # Save the resulting image
            save_path = os.path.join(target_folder, filename)
            img_upsampled.save(save_path)

if __name__ == "__main__":
    source_folder = './images/train2017'
    target_folder = './blur'
    resize_images(source_folder, target_folder)
