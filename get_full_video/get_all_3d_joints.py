import numpy as np
import cv2
from gradio_client import Client, handle_file
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mediapipe as mp
from PIL import Image
import matplotlib.animation as animation

# Helper: Compute Angles
def compute_angle(P1, P2, P3):
    """
    Computes the angle (in degrees) between vectors (P1 -> P2) and (P3 -> P2).
    
    Parameters:
    - P1, P2, P3: 3D coordinates of the points forming the angle.
    
    Returns:
    - Angle in degrees.
    """
    vec1 = P1 - P2
    vec2 = P3 - P2

    # Normalize vectors
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)
    
    if vec1_norm < 1e-6 or vec2_norm < 1e-6:
        return 0  # Return 0 for degenerate cases
        
    vec1 = vec1 / vec1_norm
    vec2 = vec2 / vec2_norm

    # Compute angle using dot product
    dot_product = np.dot(vec1, vec2)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))  # Clip for stability
    return np.degrees(angle_rad)

# Helper to display angle arc on plot
def draw_angle_arc(ax, P1, P2, P3, color='red', radius=10):
    """
    Draws an arc representing the angle at P2 formed by vectors (P1 -> P2) and (P3 -> P2).
    
    Parameters:
    - ax: 3D plot axis
    - P1, P2, P3: 3D coordinates of the points forming the angle
    - color: Color of the arc
    - radius: Size of the arc
    """
    vec1 = P1 - P2
    vec2 = P3 - P2

    # Normalize vectors
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)
    
    if vec1_norm < 1e-6 or vec2_norm < 1e-6:
        # Skip if vectors are too small
        return
        
    vec1 = vec1 / vec1_norm
    vec2 = vec2 / vec2_norm

    # Cross product gives the rotation axis
    cross = np.cross(vec1, vec2)
    cross_norm = np.linalg.norm(cross)
    
    # If the vectors are parallel (or nearly so), use another approach
    if cross_norm < 1e-6:
        # If vectors are nearly parallel, skip drawing or use a different approach
        return

    cross = cross / cross_norm
    
    # Dot product gives the cosine of the angle
    dot_product = np.dot(vec1, vec2)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    
    # Generate points for the arc
    num_points = 20
    arc_points = []
    
    # Create a proper 3D arc by rotating vec1 around the rotation axis
    for t in np.linspace(0, angle_rad, num_points):
        # Use Rodrigues rotation formula to rotate vec1 around cross axis
        cos_t = np.cos(t)
        sin_t = np.sin(t)
        rotated = vec1 * cos_t + np.cross(cross, vec1) * sin_t + cross * np.dot(cross, vec1) * (1 - cos_t)
        
        # Position at P2 with the specified radius
        arc_points.append(P2 + rotated * radius)

    arc_points = np.array(arc_points)
    
    # Draw the arc
    if len(arc_points) >= 2:  # Ensure there are enough points to draw
        return ax.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2], color=color, linewidth=2)[0]
    return None

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
    
    # interpolation for the right shoulder
    joint_3d[mp_pose.PoseLandmark.RIGHT_SHOULDER.value, 2] = joint_3d[mp_pose.PoseLandmark.LEFT_SHOULDER.value, 2] + (joint_3d[mp_pose.PoseLandmark.RIGHT_HIP.value, 2] - joint_3d[mp_pose.PoseLandmark.LEFT_HIP.value, 2])
    
    return joint_3d

def calculate_angles(joint_3d):
    """
    Calculate all angles for a single frame.
    
    Parameters:
    - joint_3d: 3D joint coordinates for a single frame
    
    Returns:
    - Dictionary with angle names and values
    """
    # Define joint angle pairs (joint_index, bone_start, bone_end)
    joint_angle_pairs = [
        (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_WRIST),
        (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_WRIST),
        (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_ANKLE),
        (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_ANKLE),
        (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_KNEE),
        (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_KNEE),
        # Shoulder angles:
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_ELBOW),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_ELBOW),
    ]
    
    angles_dict = {}
    
    for joint_index, bone_start, bone_end in joint_angle_pairs:
        P1 = joint_3d[bone_start.value]
        P2 = joint_3d[joint_index.value]  # Vertex joint
        P3 = joint_3d[bone_end.value]
        
        angle = compute_angle(P1, P2, P3)
        angles_dict[joint_index.name] = angle
        
    return angles_dict

def animate(frame_num, joints_all_frames, angles_all_frames):
    """
    Animation function to update the 3D plot for each frame.
    
    Parameters:
    - frame_num: Current frame number
    - joints_all_frames: 3D joint coordinates for all frames
    - angles_all_frames: Angle measurements for all frames
    
    Returns:
    - Updated artists for the animation
    """
    ax.clear()
    
    # Get current frame data
    joint_3d = joints_all_frames[frame_num]
    angles_dict = angles_all_frames[frame_num]
    
    # Define skeleton connections
    connections = [
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
        (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
        (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
        (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
        (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
        (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
        (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
        (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
    ]
    
    # Define joint angle pairs
    joint_angle_pairs = [
        (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_WRIST),
        (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_WRIST),
        (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_ANKLE),
        (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_ANKLE),
        (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_KNEE),
        (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_KNEE),
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_ELBOW),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_ELBOW),
    ]
    
    # Create a set of all joints we want to display (only those in connections)
    relevant_joints = set()
    for i, j in connections:
        relevant_joints.add(i.value)
        relevant_joints.add(j.value)
    
    # Draw arcs and add labels for angles
    artists = []
    radius = 20  # Adjust based on your coordinate scale
    
    for joint_index, bone_start, bone_end in joint_angle_pairs:
        P1 = joint_3d[bone_start.value]
        P2 = joint_3d[joint_index.value]
        P3 = joint_3d[bone_end.value]
        
        # Draw the arc
        arc = draw_angle_arc(ax, P1, P2, P3, color='red', radius=radius)
        if arc:
            artists.append(arc)
        
        # Get joint name and angle
        joint_name = joint_index.name.split('_')[-1]
        angle = angles_dict[joint_index.name]
        
        # Add text label
        vec1 = P1 - P2
        vec2 = P3 - P2
        vec_sum = vec1 + vec2
        norm = np.linalg.norm(vec_sum)
        if norm > 1e-6:
            vec_sum = vec_sum / norm * radius * 1.5
            text_pos = P2 + vec_sum
            text = ax.text(text_pos[0], text_pos[1], text_pos[2], 
                    f"{joint_name}: {angle:.1f}°",
                    color='black', fontsize=8, backgroundcolor='white', 
                    ha='center', va='center')
            artists.append(text)
    
    # Plot skeleton connections
    line_artists = []
    for i, j in connections:
        start_joint = joint_3d[i.value]
        end_joint = joint_3d[j.value]
        line, = ax.plot([start_joint[0], end_joint[0]], 
                       [start_joint[1], end_joint[1]], 
                       [start_joint[2], end_joint[2]], 'b', linewidth=2)
        line_artists.append(line)
    
    artists.extend(line_artists)
    
    # Plot relevant joint positions
    angle_joints = [landmark.value for landmark, _, _ in joint_angle_pairs]
    
    for i in relevant_joints:
        joint = joint_3d[i]
        if i in angle_joints:
            # Joints with angles
            scatter = ax.scatter(joint[0], joint[1], joint[2], color='green', s=30)
        else:
            # Other joints
            scatter = ax.scatter(joint[0], joint[1], joint[2], color='blue', s=15)
        artists.append(scatter)
    
    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Depth (Z)")
    ax.set_title(f"Frame {frame_num}: 3D Pose with Joint Angles")
    
    # Set consistent bounds for animation
    relevant_joint_coords = joint_3d[list(relevant_joints)]
    x_limits = [np.min(relevant_joint_coords[:, 0]), np.max(relevant_joint_coords[:, 0])]
    y_limits = [np.min(relevant_joint_coords[:, 1]), np.max(relevant_joint_coords[:, 1])]
    z_limits = [np.min(relevant_joint_coords[:, 2]), np.max(relevant_joint_coords[:, 2])]

    # Add padding
    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    
    x_limits = [x_limits[0] - 0.05 * x_range, x_limits[1] + 0.05 * x_range]
    y_limits = [y_limits[0] - 0.05 * y_range, y_limits[1] + 0.05 * y_range]
    z_limits = [z_limits[0] - 0.05 * z_range, z_limits[1] + 0.05 * z_range]

    ax.set_xlim(x_limits[::-1])
    ax.set_ylim(y_limits)
    ax.set_zlim(z_limits)
    
    ax.set_box_aspect([np.ptp(x_limits), np.ptp(y_limits), np.ptp(z_limits)])
    ax.view_init(elev=15, azim=130)
    
    return artists

# Main execution
if __name__ == "__main__":
    # Get the depth data using the API
    client = Client("depth-anything/Video-Depth-Anything")
    result = client.predict(
        input_video={"video": handle_file('input_vid.mp4')},
        max_len=500,
        target_fps=15,
        max_res=1280,
        api_name="/infer_video_depth"
    )

    depth_path = result[1]["video"]
    actual_path = result[0]["video"]

    # Load depth frames
    cap = cv2.VideoCapture(depth_path)
    depth_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        depth_frames.append(frame)
    cap.release()
    depth_frames = np.array(depth_frames)
    grey_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in depth_frames]

    # Load actual frames
    cap = cv2.VideoCapture(actual_path)
    actual_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        actual_frames.append(frame)
    cap.release()
    actual_frames = np.array(actual_frames)

    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Process all frames to get 3D joints - don't limit to 28 frames
    print("Processing frames...")
    joints_all_frames = []
    for i in range(min(len(actual_frames), len(grey_frames))):
        joints_all_frames.append(get_single_frame(i))
        print(f"Frame {i} 3D joints processed")
    joints_all_frames = np.array(joints_all_frames)

    # Save the 3D joint data as frames_all.npy
    np.save("frames_all.npy", joints_all_frames)
    print("Saved 3D joint data to frames_all.npy")

    # Calculate angles for all frames
    print("Calculating angles...")
    angles_all_frames = []
    for i, joints in enumerate(joints_all_frames):
        angles = calculate_angles(joints)
        angles_all_frames.append(angles)
        print(f"Frame {i} angles calculated")
    
    # Save angles data (convert dict to structured format for saving)
    angle_names = list(angles_all_frames[0].keys())
    angles_array = np.zeros((len(angles_all_frames), len(angle_names)))
    
    for i, angles in enumerate(angles_all_frames):
        for j, name in enumerate(angle_names):
            angles_array[i, j] = angles[name]
    
    np.save("angles_all_frames.npy", angles_array)
    with open("angle_names.txt", "w") as f:
        for name in angle_names:
            f.write(name + "\n")
    print("Saved angle data to angles_all_frames.npy and angle_names.txt")

    # Create animation
    print("Creating animation...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    anim = animation.FuncAnimation(
        fig, animate, frames=len(joints_all_frames),
        fargs=(joints_all_frames, angles_all_frames),
        interval=100, blit=True)

    # Save animation (uncomment to save)
    # anim.save('skeleton_animation.mp4', writer='ffmpeg', fps=10)

    plt.show()