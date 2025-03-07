import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mediapipe as mp
from transformers import DPTForDepthEstimation, DPTImageProcessor
from PIL import Image

# combines joint locatio nand depth 
# Load MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
print("ready")

# --- Step 1: Load Image ---

image = np.load("steph_actual.npy")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pose.process(image_rgb)
if not results.pose_landmarks:
    raise ValueError("No human detected!")

# Get 2D Joint Positions (in pixel coordinates)
pose_landmarks = results.pose_landmarks.landmark
joint_2d = np.array([[lm.x * image.shape[1], lm.y * image.shape[0]] for lm in pose_landmarks])
joint_2d = np.clip(joint_2d, 0, [image.shape[1] - 1, image.shape[0] - 1])  # Clip to image bounds

# --- Step 2: Load Depth Map from NPY ---
depth_map = np.load("steph_depth.npy")  # Load depth map
print(image.shape)
print(depth_map.shape)
if depth_map.shape != (image.shape[0], image.shape[1]):
    raise ValueError("Depth map resolution does not match the image resolution!")

# --- Step 3: Extract Depth for Each Joint ---
joint_3d = []
for x, y in joint_2d:
    x, y = int(x), int(y)
    z = depth_map[y, x]  # Get depth at joint location
    joint_3d.append([x, y, z])
joint_3d = np.array(joint_3d)


joint_3d[mp_pose.PoseLandmark.RIGHT_SHOULDER.value, 2] = joint_3d[mp_pose.PoseLandmark.LEFT_SHOULDER.value, 2] + (joint_3d[mp_pose.PoseLandmark.RIGHT_HIP.value, 2] - joint_3d[mp_pose.PoseLandmark.LEFT_HIP.value, 2])

# Define the selected connections between landmarks
connections = [
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.LEFT_SHOULDER),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.LEFT_HIP),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
    (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_KNEE),
    (mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_KNEE),
]

# --- Step 4: Plot the 3D Pose ---
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot only the connections (no extra joints)
for i, j in connections:
    start_joint = joint_3d[i.value]
    end_joint = joint_3d[j.value]
    
    ax.plot([start_joint[0], end_joint[0]], 
            [start_joint[1], end_joint[1]], 
            [start_joint[2], end_joint[2]], 'b')

# Plot nose separately in green
nose_index = mp_pose.PoseLandmark.NOSE
nose_3d = joint_3d[nose_index.value]
ax.scatter(nose_3d[0], nose_3d[1], nose_3d[2], c='g', marker='^', s=100, label="Nose")

# Labels & View
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Depth (Z)")

# Ensure equal scaling across all axes
x_limits = [np.min(joint_3d[:, 0]), np.max(joint_3d[:, 0])]
y_limits = [np.min(joint_3d[:, 1]), np.max(joint_3d[:, 1])]
z_limits = [np.min(joint_3d[:, 2]), np.max(joint_3d[:, 2])]

ax.set_xlim(x_limits[::-1])
ax.set_ylim(y_limits)
ax.set_zlim(z_limits)

# Force equal aspect ratio
ax.set_box_aspect([np.ptp(x_limits), np.ptp(y_limits), np.ptp(z_limits)])

ax.view_init(elev=15, azim=130)
ax.legend()
plt.show()
