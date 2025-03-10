import numpy as np
import cv2
from gradio_client import Client, handle_file
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mediapipe as mp
from PIL import Image



def get_single_frame(frame_num):
    image = actual_frames[frame_num]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if not results.pose_landmarks:
        raise ValueError("No human detected!")

    # Get 2D Joint Positions (in pixel coordinates)
    pose_landmarks = results.pose_landmarks.landmark
    joint_2d = np.array([[lm.x * image.shape[1], lm.y * image.shape[0]] for lm in pose_landmarks])
    joint_2d = np.clip(joint_2d, 0, [image.shape[1] - 1, image.shape[0] - 1])  # Clip to image bounds

    # --- Step 2: Load Depth Map from NPY ---
    depth_map = grey_frames[frame_num]
    if depth_map.shape != (image.shape[0], image.shape[1]):
        raise ValueError("Depth map resolution does not match the image resolution!")

    # --- Step 3: Extract Depth for Each Joint ---
    joint_3d = []
    for x, y in joint_2d:
        x, y = int(x), int(y)
        z = depth_map[y, x]  # Get depth at joint location
        joint_3d.append([x, y, z])
    joint_3d = np.array(joint_3d)


    # interpolation for the right shoulder, you might want to get rid of this line if right shoulder is not blocked
    # or might want to do interpolation in some different way 
    joint_3d[mp_pose.PoseLandmark.RIGHT_SHOULDER.value, 2] = joint_3d[mp_pose.PoseLandmark.LEFT_SHOULDER.value, 2] + (joint_3d[mp_pose.PoseLandmark.RIGHT_HIP.value, 2] - joint_3d[mp_pose.PoseLandmark.LEFT_HIP.value, 2])
    print(type(joint_3d))
    return joint_3d




# main
client = Client("depth-anything/Video-Depth-Anything")
result = client.predict(
        input_video={"video":handle_file('input_vid.mp4')},
        max_len=500,
        target_fps=15,
        max_res=1280,
        api_name="/infer_video_depth"
)


depth_path = result[1]["video"]
actual_path = result[0]["video"]

cap = cv2.VideoCapture(depth_path)

depth_frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop when video ends
    depth_frames.append(frame)  # Append the frame as a NumPy array

cap.release()

depth_frames = np.array(depth_frames)  # Convert list to NumPy array
grey_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in depth_frames]


cap = cv2.VideoCapture(actual_path)
actual_frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop when video ends
    actual_frames.append(frame)  # Append the frame as a NumPy array

cap.release()

actual_frames = np.array(actual_frames)  # Convert list to NumPy array
# combines joint locatio nand depth 
# Load MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


joints_all_frames = []
for i in range(0,28):
    joints_all_frames.append(get_single_frame(i))
    print(f"frame {i} done")

joints_all_frames = np.array(joints_all_frames)
np.save("frames_all", joints_all_frames)