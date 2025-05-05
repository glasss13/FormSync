import numpy as np


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