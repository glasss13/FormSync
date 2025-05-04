import os
import cv2

import torch
    
from pose2d import get_pose2d
from pose3d import get_pose3D

VIDEO_PATH: str = "./klay.mp4"


if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")



def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    keypoints = get_pose2d(VIDEO_PATH)
    get_pose3D(VIDEO_PATH, keypoints)

    

if __name__ == "__main__":
    main()
