import os
import cv2

import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import numpy as np
    
from pose2d import get_pose2d
from pose3d import get_pose3D, show3Dpose

VIDEO_PATH: str = "./assets/klay.mp4"

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    keypoints = get_pose2d(VIDEO_PATH)
    joint_coordinates = get_pose3D(VIDEO_PATH, keypoints)

    print(joint_coordinates)

    for x in joint_coordinates:

        fig = plt.figure(figsize=(9.6, 5.4))
        gs = gridspec.GridSpec(1, 1)
        gs.update(wspace=-0.00, hspace=0.05) 
        ax = plt.subplot(gs[0], projection='3d')
        show3Dpose(x, ax)
        
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height(physical=True)

        buffer = fig.canvas.buffer_rgba()

        img_rgba = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 4)

        img_bgr = cv2.cvtColor(img_rgba.copy(), cv2.COLOR_RGBA2BGR)

        # --- Display with OpenCV ---
        cv2.imshow('Matplotlib 3D Pose via OpenCV', img_bgr)
        cv2.waitKey(0)


    plt.clf()
    plt.close(fig)


if __name__ == "__main__":
    main()


# things the backend needs to do:
# 1. compute joint coordinates for each frame
# 2. compute loss score based on the coordinates and the selected reference video
# 3. Send down the reference video with joints overlaid
# 4. Generate a video of the user's inputted video with joints overlaid
# 5. Generate a video of the user's inputted video with skeleton in 3d space.
