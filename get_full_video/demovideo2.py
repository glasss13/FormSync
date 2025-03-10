import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import mediapipe as mp

def smooth(array):
    threshold = 50
    
    # Loop through each joint
    for joint in range(array.shape[1]):
        # Loop through each dimension (x, y, z)
        for dim in [2]:  # 3 dimensions: 0 for x, 1 for y, 2 for z
            # Loop through each 3D point for that joint across time (skip first and last frame)
            for time in range(1, array.shape[0] - 1):  # Skip first and last frame (we need two surrounding points)
                # Calculate the difference between consecutive frames in the specific dimension
                diff = np.abs(array[time+1, joint, dim] - array[time-1, joint, dim])
                
                # If the difference in any dimension exceeds the threshold, it's an outlier
                if diff > threshold:
                    print("HIT")
                    # Replace with the average of the surrounding frames
                    array[time, joint, dim] = (array[time-1, joint, dim] + array[time+1, joint, dim]) / 2
    
    return array

def compute_angle(P1, P2, P3):
    """
    Computes the angle (in degrees) between vectors (P1 -> P2) and (P3 -> P2).
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

def draw_angle_arc(ax, P1, P2, P3, color='red', radius=10):
    """
    Draws an arc representing the angle.
    """
    vec1 = P1 - P2
    vec2 = P3 - P2

    # Normalize vectors
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)
    
    if vec1_norm < 1e-6 or vec2_norm < 1e-6:
        return
        
    vec1 = vec1 / vec1_norm
    vec2 = vec2 / vec2_norm

    # Cross product gives the rotation axis
    cross = np.cross(vec1, vec2)
    cross_norm = np.linalg.norm(cross)
    
    if cross_norm < 1e-6:
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
        cos_t = np.cos(t)
        sin_t = np.sin(t)
        rotated = vec1 * cos_t + np.cross(cross, vec1) * sin_t + cross * np.dot(cross, vec1) * (1 - cos_t)
        arc_points.append(P2 + rotated * radius)

    arc_points = np.array(arc_points)
    
    if len(arc_points) >= 2:
        ax.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2], color=color, linewidth=2)

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

# Define joint angle pairs (which joints to calculate angles for)
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

# Function to get better display name for joint
def get_joint_display_name(joint_name):
    # Split into parts and get the side (LEFT/RIGHT) and the joint name
    parts = joint_name.split('_')
    if len(parts) > 1:
        side = parts[0]
        joint = parts[-1]
        return f"{side.title()} {joint.title()}"
    return joint_name

# Load frames from files
frames = np.load("frames_all.npy")
print("Loaded frames shape:", frames.shape)

# Swap Y and Z axes to make the skeleton stand upright
frames[:, :, [1, 2]] = frames[:, :, [2, 1]]

frames = smooth(frames)

# Set up the figure and axis
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Add a separate text area for angle values
angle_text = fig.text(0.02, 0.5, "", transform=fig.transFigure, 
                      verticalalignment='center', fontsize=10)

# Add frame counter in the upper right
frame_counter = fig.text(0.85, 0.95, "", transform=fig.transFigure, 
                         verticalalignment='top', fontsize=12)

# Set the fixed axis limits once - but now with the axes swapped
x_min = np.min(frames[:, :, 0]) - 20
x_max = np.max(frames[:, :, 0]) + 20
y_min = np.min(frames[:, :, 1]) - 20
y_max = np.max(frames[:, :, 1]) + 50  # Add padding
z_min = np.min(frames[:, :, 2]) - 20
z_max = np.max(frames[:, :, 2]) + 50  # Add padding

# Variables to track view angle - set to make Z axis vertical (after swap)
paused = False
view_elev = 90
view_azim = 270
initial_elev = view_elev  # Define initial elevation before using it

# Define the update function for the animation
def update(frame_idx):
    global view_elev, view_azim
    
    # Store current view angle if it exists
    if hasattr(ax, 'elev') and ax.elev is not None:
        view_elev = ax.elev
    if hasattr(ax, 'azim') and ax.azim is not None:
        view_azim = ax.azim
    
    ax.clear()
    joint_3d = frames[frame_idx]
    
    # Set labels - now Z is vertical
    ax.set_xlabel("X")
    ax.set_ylabel("Y")  # Now Y is depth (forward/backward)
    ax.set_zlabel("Z")  # Now Z is height (up/down)
    
    # Set consistent limits - with swapped Y and Z axes
    ax.set_xlim([x_max, x_min])  # Reversed x-axis
    ax.set_ylim([y_min, y_max])  # Now Y is depth (forward/backward)
    ax.set_zlim([z_max, z_min])  # Now Z is height (up/down) and flip
    ax.set_box_aspect([(x_max-x_min), (y_max-y_min), (z_max-z_min)])
    
    # Move the XY plane grid to the top (z_max) instead of the bottom
    ax.zaxis._axinfo['juggled'] = (2, 0, 1)  # This controls which plane is shown
    ax.zaxis.set_pane_color((0.8, 0.8, 0.8, 0.2))  # Light gray, semi-transparent
    
    # Set grid positions
    ax.xaxis.set_pane_position = {'y': y_min, 'z': z_min}  # YZ plane
    ax.yaxis.set_pane_position = {'x': x_max, 'z': z_min}  # XZ plane
    ax.zaxis.set_pane_position = {'x': x_max, 'y': y_min}  # XY plane (move to z_max)
    
    # Adjust grid to be at the top (z_max) instead of bottom (z_min)
    ax.zaxis._axinfo['grid']['color'] = (0.8, 0.8, 0.8, 0.5)
    ax.zaxis._axinfo['grid']['linestyle'] = '--'
    ax.zaxis._axinfo['grid']['linewidth'] = 0.8
    
    # Plot skeleton connections
    for start, end in connections:
        start_joint = joint_3d[start.value]
        end_joint = joint_3d[end.value]
        
        ax.plot([start_joint[0], end_joint[0]], 
                [start_joint[1], end_joint[1]],  # Y coordinates now depth
                [start_joint[2], end_joint[2]], 'b', linewidth=2)  # Z coordinates now height
    
    # Calculate angles and draw arcs
    angle_values = {}
    
    for joint_index, bone_start, bone_end in joint_angle_pairs:
        P1 = joint_3d[bone_start.value]
        P2 = joint_3d[joint_index.value]
        P3 = joint_3d[bone_end.value]
        
        # Calculate angle
        angle = compute_angle(P1, P2, P3)
        
        # Use full joint name as key
        angle_values[joint_index.name] = angle
        
        # Draw arc
        draw_angle_arc(ax, P1, P2, P3, color='red', radius=10)
    
    # Update frame counter
    frame_counter.set_text(f"Frame: {frame_idx}")
    
    # Update angle values text - group by joint type
    angle_text_str = "Angle Values:\n\n"
    
    # Group angles by joint type (Elbow, Knee, Hip, Shoulder)
    joint_types = {'ELBOW': [], 'KNEE': [], 'HIP': [], 'SHOULDER': []}
    
    for joint_name, angle in angle_values.items():
        # Determine joint type
        for joint_type in joint_types.keys():
            if joint_type in joint_name:
                joint_types[joint_type].append((joint_name, angle))
                break
    
    # Display angles by group
    for joint_type, angles in joint_types.items():
        if angles:
            angle_text_str += f"{joint_type}:\n"
            for joint_name, angle in angles:
                display_name = get_joint_display_name(joint_name)
                angle_text_str += f"  {display_name}: {angle:.1f}°\n"
            angle_text_str += "\n"
    
    angle_text.set_text(angle_text_str)
    
    # Restore the view angle
    ax.view_init(elev=view_elev, azim=view_azim)

# Add callback to make the plot pausable
def on_key_press(event):
    global paused
    if event.key == 'p':
        if not paused:
            ani.pause()  # Pause animation
            paused = True
        else:
            ani.resume()  # Resume animation
            paused = False
    elif event.key == 'r':
        # Reset view
        global view_elev, view_azim
        view_elev = 90  # Reset to view with Z as vertical
        view_azim = 270
        ax.view_init(elev=view_elev, azim=view_azim)
        fig.canvas.draw()

fig.canvas.mpl_connect('key_press_event', on_key_press)

def on_mouse_move(event):
    global view_azim, view_elev

    if event.inaxes == ax and event.button == 1:  # Left mouse button pressed
        if hasattr(event, 'x') and hasattr(event, 'y') and event.x is not None and event.y is not None:
            dx = event.x - on_mouse_move.prev_x  # Horizontal movement
            dy = event.y - on_mouse_move.prev_y  # Vertical movement
            
            # Rotate around the Y-axis by adjusting azimuth only
            view_azim += dx * 0.5  # Horizontal drag rotates around Y-axis
            view_elev = 0  # Keep the camera level (XZ plane is stable)

            ax.view_init(elev=view_elev, azim=view_azim)
            fig.canvas.draw_idle()

        on_mouse_move.prev_x = event.x  # Store previous x position
        on_mouse_move.prev_y = event.y  # Store previous y position

on_mouse_move.prev_x = 0
on_mouse_move.prev_y = 0
view_elev = 0  # Keep the camera at a level angle
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

# Initial view rotation - Z as vertical axis
ax.view_init(elev=view_elev, azim=view_azim)

# Create the animation object
ani = FuncAnimation(fig, update, frames=len(frames), interval=200, repeat=True)

# Display the animation
plt.tight_layout()
plt.show()