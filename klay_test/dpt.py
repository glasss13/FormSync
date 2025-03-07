import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mediapipe as mp
from transformers import DPTForDepthEstimation, DPTImageProcessor
from PIL import Image

# Load DPT Model (Depth Anything)
model_name = "Intel/dpt-large"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dpt_model = DPTForDepthEstimation.from_pretrained(model_name).to(device)
dpt_processor = DPTImageProcessor.from_pretrained(model_name)

# Load MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
print("ready")

# --- Step 1: Load Image ---
image_path = "steph2.jpg"  # Change this to your image path
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("done 1")

# --- Step 2: Run MediaPipe Pose ---
results = pose.process(image_rgb)
if not results.pose_landmarks:
    raise ValueError("No human detected!")
print("done 2")

# Get 2D Joint Positions (in pixel coordinates)
pose_landmarks = results.pose_landmarks.landmark
joint_2d = np.array([[lm.x * image.shape[1], lm.y * image.shape[0]] for lm in pose_landmarks])
joint_2d = np.clip(joint_2d, 0, [image.shape[1]-1, image.shape[0]-1])  # Clip to image bounds

# --- Step 3: Run DPT for Depth Estimation ---
input_image = Image.fromarray(image_rgb)
inputs = dpt_processor(images=input_image, return_tensors="pt").to(device)
with torch.no_grad():
    depth_map = dpt_model(**inputs).predicted_depth.squeeze().cpu().numpy()

# Resize depth map to match image size
depth_map_resized = cv2.resize(depth_map, (image.shape[1], image.shape[0]))

# --- Step 4: Extract Depth for Each Joint ---
joint_3d = []
for x, y in joint_2d:
    x, y = int(x), int(y)
    z = depth_map_resized[y, x]  # Get depth at joint location
    joint_3d.append([x, y, z])
joint_3d = np.array(joint_3d)

# Define the selected connections between landmarks
connections = [
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.LEFT_SHOULDER),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
]

# --- Step 5: Plot the 3D Pose ---
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
ax.view_init(elev=15, azim=130)
ax.legend()
plt.show()

# --- Step 6: Overlay Connections and Nose on Original Image ---
overlay_image = image.copy()  # Make a copy to draw on

# Draw only the selected connections
for i, j in connections:
    pt1 = tuple(map(int, joint_2d[i.value]))
    pt2 = tuple(map(int, joint_2d[j.value]))
    cv2.line(overlay_image, pt1, pt2, (255, 0, 0), thickness=2)  # Blue lines for connections

# Draw nose separately
nose_2d = joint_2d[nose_index.value]
cv2.circle(overlay_image, tuple(map(int, nose_2d)), radius=5, color=(0, 255, 0), thickness=-1)  # Green for nose

# Display the overlaid image
plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct colors
plt.axis("off")
plt.title("Detected Joints Overlaid on Image")
plt.show()
