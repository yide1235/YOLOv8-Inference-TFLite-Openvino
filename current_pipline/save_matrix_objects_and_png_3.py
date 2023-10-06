import cv2
import os
import numpy as np
import time

def is_image_blurry(frame, threshold=100):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute the Laplacian variance of the frame
    laplacian_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()

    # Check if the variance is below the threshold
    if laplacian_var < threshold:
        return True
    else:
        return False

def save_frames_from_raw_video(video_path, width, height, output_dir):
    


    # Calculate the size of one frame in bytes (assuming BGRx format, 4 bytes per pixel)
    frame_size_bytes = width * height * 4

    # Open the raw video file in binary mode
    with open(video_path, 'rb') as file:
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        frames = []
        frame_number = 0
        while True:
            # Read the frame data from the raw video file
            frame_data = file.read(frame_size_bytes)

            # Break the loop if there is no data left (end of video)
            if not frame_data:
                break

            # Convert the frame data to an OpenCV matrix object
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape((height, width, 4))  # Assuming BGRx format (4 channels)
            frame = frame[:, :, :3]  # Discard the alpha channel if it exists

            # Add the frame to the list
            frames.append(frame)

            frame_number += 1

    print(f"{frame_number} frames read successfully.")

    # Calculate the average frame
    average_frame = np.mean(frames, axis=0).astype(np.uint8)

    # Find the frame with the most difference from the average frame that is not blurry
    max_difference_frame = None
    max_difference = 0
    for frame in frames:
        if not is_image_blurry(frame, blurry_threshold):
            difference = np.sum(np.abs(frame - average_frame))
            if difference > max_difference:
                max_difference_frame = frame
                max_difference = difference
    

    print("execute time =")
    print(timestamp2 - timestamp1)
    # Save the average frame as an image in the output directory
    average_frame_filename = os.path.join(output_dir, "average_frame.png")
    cv2.imwrite(average_frame_filename, average_frame)

    # Save the frame with the most difference as an image in the output directory
    if max_difference_frame is not None:
        max_difference_frame_filename = os.path.join(output_dir, "max_difference_frame.png")
        cv2.imwrite(max_difference_frame_filename, max_difference_frame)
        print("Average frame and frame with most difference (not blurry) saved successfully.")
    else:
        print("No frame found with the most difference (not blurry).")

if __name__ == "__main__":
    # Replace 'path_to_your_video.raw' with the actual path to your raw BGRx video file
    video_path = "/home/root/farid/gst-test/videos/output00015.raw"
    blurry_threshold = 300  # Adjust this threshold as needed

    # Set the width and height of your raw video
    width = 480
    height = 544

    # Replace 'output_directory' with the directory where you want to save the frames
    output_directory = "frames_output"

    save_frames_from_raw_video(video_path, width, height, output_directory)
