import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(2)

# Landmarks to plot
landmark_ids = [
    mp_pose.PoseLandmark.NOSE,
    mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE
]

# Connections to draw lines between
connections = [
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
    (mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_KNEE),
    (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_KNEE)
]

def plot_3d_landmarks(landmarks):
    """Plots the selected landmarks in 3D space with lines, normalized by hip position."""
    
    # Get hip positions
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    # Compute hip center
    hip_center_x = (left_hip.x + right_hip.x) / 2
    hip_center_y = (left_hip.y + right_hip.y) / 2
    hip_center_z = (left_hip.z + right_hip.z) / 2

    # Extract normalized landmark positions
    x_vals, y_vals, z_vals = [], [], []
    for lm_id in landmark_ids:
        lm = landmarks[lm_id]
        x_vals.append(lm.x - hip_center_x)
        y_vals.append(lm.y - hip_center_y)
        z_vals.append(lm.z - hip_center_z)

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    ax.scatter(x_vals, z_vals, -np.array(y_vals), c='blue', marker='o')

    # Draw connections
    for joint1, joint2 in connections:
        lm1, lm2 = landmarks[joint1], landmarks[joint2]
        x_pair = [lm1.x - hip_center_x, lm2.x - hip_center_x]
        y_pair = [lm1.y - hip_center_y, lm2.y - hip_center_y]
        z_pair = [lm1.z - hip_center_z, lm2.z - hip_center_z]
        ax.plot(x_pair, z_pair, [-y for y in y_pair], c='black')

    # Ensure equal scale across all axes
    max_range = max(np.ptp(x_vals), np.ptp(y_vals), np.ptp(z_vals)) / 2
    mid_x, mid_y, mid_z = np.mean(x_vals), np.mean(y_vals), np.mean(z_vals)

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_z - max_range, mid_z + max_range)
    ax.set_zlim(-mid_y - max_range, -mid_y + max_range)

    # Labels
    ax.set_xlabel("X")
    ax.set_ylabel("Z (Depth)")
    ax.set_zlabel("Y (Height)")
    ax.set_title("3D Pose Landmarks")

    plt.show()

i = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Draw landmarks if detected
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get skeletal points
        print(i)

        # Plot 3D landmarks at frame 150
        if i == 150:
            plot_3d_landmarks(results.pose_landmarks.landmark)
        i += 1

    # Display frame
    cv2.imshow('MediaPipe Pose', frame)

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
