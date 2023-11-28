# import cv2

# # Open the first video
# video1 = cv2.VideoCapture('./367.mp4')

# # Open the second video
# video2 = cv2.VideoCapture('./367_stereo.mp4')

# # Set the frame you want to grab (frame number starts from 0)
# frame_to_grab = 1

# # Function to extract a specific frame from a video
# def extract_frame(video, frame_number):
#     video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
#     ret, frame = video.read()
#     return frame

# # Extract frame 20 from the first video
# frame_from_video1 = extract_frame(video1, frame_to_grab)

# # Extract frame 20 from the second video
# frame_from_video2 = extract_frame(video2, frame_to_grab)

# # Release the video captures
# video1.release()
# video2.release()

# # Save the frames as images (optional)
# cv2.imwrite('./left/frame1_from_video1.jpg', frame_from_video1)
# cv2.imwrite('./right/frame1_from_video2.jpg', frame_from_video2)


import cv2

# Open the first video
video1 = cv2.VideoCapture('./367.mp4')

# Open the second video
video2 = cv2.VideoCapture('./367_stereo.mp4')

# Check if both videos are opened successfully
if not video1.isOpened() or not video2.isOpened():
    print("Error opening videos")
    exit()

# Frame counter
frame_counter = 0

while True:
    # Read frame from both videos
    ret1, frame_from_video1 = video1.read()
    ret2, frame_from_video2 = video2.read()

    # Check if frames are read successfully
    if not ret1 or not ret2:
        print("Finished reading frames.")
        break

    # Save the frames as images
    cv2.imwrite(f'./left/frame{frame_counter}.jpg', frame_from_video1)
    cv2.imwrite(f'./right/frame{frame_counter}.jpg', frame_from_video2)

    # Increment the frame counter
    frame_counter += 1

# Release the video captures
video1.release()
video2.release()