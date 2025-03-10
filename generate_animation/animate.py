import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Define body part connections
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

# Load frames from files
frames = []
for name in ["joints_10.npy", "joints_17.npy", "joints_20.npy", "joints_27.npy"]:
    joint_3d = np.load(name)
    frames.append(joint_3d)

# Set up the figure and axis (only once)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_box_aspect([1, 1, 1])  # Forces equal scaling for all axes

# Set the fixed axis limits once (optional, adjust as needed)
x_limits = [np.min(frames[0][:, 0])-20, np.max(frames[0][:, 0]+20)]
y_limits = [np.min(frames[0][:, 1])-50, np.max(frames[0][:, 1])+20]
z_limits = [np.min(frames[0][:, 2])-20, np.max(frames[0][:, 2])+20]
ax.set_xlim(x_limits[::-1])
ax.set_ylim(y_limits)
ax.set_zlim(z_limits)
ax.set_box_aspect([np.ptp(x_limits), np.ptp(y_limits), np.ptp(z_limits)])

# Define the update function for the animation
def update(frame):
    joint_3d = frames[frame]
    ax.cla()  # Clear the previous frame
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Plot only the connections (no extra joints)
    for i, j in connections:
        start_joint = joint_3d[i.value]
        end_joint = joint_3d[j.value]
        
        ax.plot([start_joint[0], end_joint[0]], 
                [start_joint[1], end_joint[1]], 
                [start_joint[2], end_joint[2]], 'b')

    # Reapply the axis limits in every frame
    ax.set_xlim(x_limits[::-1])
    ax.set_ylim(y_limits)
    ax.set_zlim(z_limits)

# Create the animation object
ani = FuncAnimation(fig, update, frames=len(frames), interval=600)

# Display the animation
plt.show()
