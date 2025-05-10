import numpy as np


#Function to compute the angle between 3 points
def compute_angle(p1, p2, p3):
    """
    Computes the angle (in degrees) between vectors (P1 -> P2) and (P3 -> P2).
  
    Parameters:
    - P1, P2, P3: 3D coordinates of the points forming the angle.
  
    Returns:
   - Angle in degrees.
    """
    vec1 = p1 - p2
    vec2 = p3 - p2

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

def extract_angles_from_frame(joint_3d):
    """
    Compute joint angles at frame frame_idx from pose3d_data.

    Parameters:
    -joint_3d (np.ndarray): shape (17, 3)

    Returns: 
    - angles_dict (dict): keys are (joint_a, vertex_joint, joint_b), values are angles in degrees. 
    """
    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]
    
    from collections import defaultdict 
    
    #build adjacent list
    neighbors = defaultdict(list)
    for a, b in connections:
        neighbors[a].append(b)
        neighbors[b].append(a)
    
    angles_dict = {}
    for center_joint, connected in neighbors.items():
        if len(connected) < 2:
            continue
        for i in range(len(connected)):
            for j in range(i +1, len(connected)):
                u, v = connected[i], connected[j]
                p1, p2, p3 = joint_3d[u], joint_3d[center_joint], joint_3d[v]
                angle = compute_angle(p1, p2, p3)
                u, v = min(u, v), max(u, v)
                key = tuple([u, center_joint, v])  # e.g., (4, 6, 12)
                angles_dict[key] = angle

    return angles_dict


def get_min_angle_diff(selected_angles, target_angles, search_width):

    '''
    given a list of angles corresponding to a single frame and a list of list of angles corresponding to the target video,
    returns the angle differences of the target frame with the minimum mse of the angles

    assumes selected_angles is of form [angle1, angle2, ...]
    and target_angles is of form [[angle1, angle2,...], [angle1, angle2,...], ...]

    search_width should be odd
    '''

    assert search_width % 2 == 1, "search_width must be an odd number"
    half_width = (search_width - 1) // 2
    min_mse = float('inf')
    best_diff = None

    for i in range(half_width, len(target_angles) - half_width):
        candidate = target_angles[i]

        # Compute minimal angular difference using modular arithmetic
        diff_array = np.array(candidate) - np.array(selected_angles)
        diff_array = (diff + 180) % 360 - 180  # Ensures difference is in range [-180, 180]

        mse = np.mean(diff_array**2)
        if mse < min_mse:
            min_mse = mse
            best_diff = diff_array

    return (best_diff, min_mse)
