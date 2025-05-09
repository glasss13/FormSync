import os
import tempfile
from types import SimpleNamespace
import cv2
from fastdtw import fastdtw

import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import numpy as np

from hrnet.config import _C as HRNET_CONFIG, update_config
from pose2d import get_pose2d, load_hrnet_model, load_YOLO_model
from pose3d import get_pose3D, show3Dpose, init_model as init_poseformer_model
from utils2 import extract_angles_from_frame

from flask import Flask, request, abort, jsonify

app = Flask(__name__)

POSEFORMER_ARGS = SimpleNamespace()
POSEFORMER_ARGS.embed_dim_ratio, POSEFORMER_ARGS.depth, POSEFORMER_ARGS.frames = 32, 4, 243
POSEFORMER_ARGS.number_of_kept_frames, POSEFORMER_ARGS.number_of_kept_coeffs = 27, 27
POSEFORMER_ARGS.pad = (POSEFORMER_ARGS.frames - 1) // 2
POSEFORMER_ARGS.n_joints, POSEFORMER_ARGS.out_joints = 17, 17

HRNET_ARGS = SimpleNamespace()
HRNET_ARGS.cfg = "./hrnet/w48_384x288_adam_lr1e-3.yaml"
HRNET_ARGS.opts = []
HRNET_ARGS.modelDir = "./hrnet/pose_hrnet_w48_384x288.pth"
HRNET_ARGS.det_dim = 416      
HRNET_ARGS.thred_score = 0.30 
HRNET_ARGS.animation = False  
HRNET_ARGS.num_person = 1     
HRNET_ARGS.video = 'camera'   
HRNET_ARGS.gpu = '0'

REFERENCE_VIDEOS = [
    "./reference_videos/klay.mp4",
    "./reference_videos/squat.mp4"
]


def init_models():
    poseformer_model = init_poseformer_model(POSEFORMER_ARGS)
    update_config(HRNET_CONFIG, HRNET_ARGS)
    hrnet_model = load_hrnet_model(HRNET_CONFIG)
    yolo_model = load_YOLO_model()

    return {
        "poseformer": poseformer_model,
        "hrnet":  hrnet_model,
        "yolo": yolo_model
    }

MODELS = init_models()

def init_reference_videos():
    res = []
    for reference_video_path in REFERENCE_VIDEOS:
        keypoints = get_pose2d(reference_video_path, MODELS["yolo"], MODELS["hrnet"])
        joint_coordinates = get_pose3D(reference_video_path, MODELS["poseformer"], POSEFORMER_ARGS, keypoints)
        res.append(joint_coordinates)
        
    print("finished initializing reference videos")
    return res



REFERENCE_VIDEO_JOINTS = init_reference_videos()



def normalize_frame(joint_coords, reference_joint_idx=0):
    return joint_coords - joint_coords[reference_joint_idx]

def frame_distance(joint_coords_a, joint_coords_b, weights_dict = None):

    angles_a, weights = extract_angles_from_frame(joint_coords_a)
    angles_b, _ = extract_angles_from_frame(joint_coords_b)

    if weights_dict is not None:
        weights = weights_dict

    distance = 0

    for key in angles_a:
        angle_a = angles_a[key]
        angle_b = angles_b[key]
        difference = min(abs(angle_a-angle_b), 360-abs(angle_a-angle_b))
        distance += weights[key] * difference ** 2

    return distance ** 0.5


    '''
    norm_joint_coords_a = normalize_frame(joint_coords_a)
    norm_joint_coords_b = normalize_frame(joint_coords_b)

    total_distance = 0.0
    num_joints = norm_joint_coords_a.shape[0]

    for i in range(num_joints):
        # use the euclidean distance between the joints for now
        # we should probably find something better?
        total_distance += np.linalg.norm(norm_joint_coords_a[i] - norm_joint_coords_b[i])

    return total_distance
    '''

def align_videos(reference_video, user_video):
    max_frames = max(len(reference_video), len(user_video))
    _, warping_path = fastdtw(reference_video, user_video, radius=max_frames, dist=frame_distance)
    return warping_path
[


@app.route("/<int:reference_id>", methods=["POST"])
def send_video(reference_id: int):
    if reference_id >= len(REFERENCE_VIDEO_JOINTS):
        print(f"invalid reference video id: {reference_id}")
        abort(404)

    if "file" not in request.files:
        print("no video in request")
        abort(404)

    file = request.files["file"]

    if file.filename == "":
        print("no selected file")
        abort(404)

    ext = os.path.splitext(file.filename)[1]

    try:
        video_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        video_file_path = video_file.name
        file.save(video_file_path)
        video_file.close()

        keypoints = get_pose2d(video_file_path, MODELS["yolo"], MODELS["hrnet"])
        joint_coordinates = get_pose3D(video_file_path, MODELS["poseformer"], POSEFORMER_ARGS, keypoints)

        video_capture = cv2.VideoCapture(video_file_path)

        video_capture.release()

    finally:
        if video_file_path and os.path.exists(video_file_path):
            try:
                os.remove(video_file_path)
            except:
                print("error deleting temporary file {video_file_path}")

    
    reference_joint_coordinates = REFERENCE_VIDEO_JOINTS[reference_id]

    print("aligning videos...")
    path = align_videos(reference_joint_coordinates, joint_coordinates)

    fig = plt.figure(figsize=(19.2, 5.4))

    gs = gridspec.GridSpec(1, 2)
    gs.update(wspace=0.05, hspace=0.05)

    ax1 = fig.add_subplot(gs[0], projection='3d')
    ax2 = fig.add_subplot(gs[1], projection='3d')

    path = sorted(path, key=lambda x: frame_distance(reference_joint_coordinates[x[0]], joint_coordinates[x[1]]), reverse=True)

    mse_sum = 0
    for step, (ref_idx, user_idx) in enumerate(path):

        fd = frame_distance(ref_pose, user_pose)

        ref_pose = reference_joint_coordinates[ref_idx]
        user_pose = joint_coordinates[user_idx]

        print("distance: ", fd)
        mse_sum += fd

        ax1.clear()
        ax2.clear()

        show3Dpose(ref_pose, ax1)
        show3Dpose(user_pose, ax2)

        fig.canvas.draw()
        width, height = fig.canvas.get_width_height(physical=True)

        buffer = fig.canvas.buffer_rgba()

        img_rgba = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 4)

        img_bgr = cv2.cvtColor(img_rgba.copy(), cv2.COLOR_RGBA2BGR)

        cv2.imshow('Matplotlib 3D Pose via OpenCV', img_bgr)
        cv2.waitKey(0)
    
    plt.clf()
    plt.close(fig)

    response = {
        "joint_coordinates": joint_coordinates,
        "avg_mse_sum": mse_sum / len(path)
    }

    return jsonify(response), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)

VIDEO_PATH: str = "./assets/squat2.mp4"

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    models = init_models()

    # poseformer_model, hrnet_model, yolo_model = init_models()

    keypoints = get_pose2d(VIDEO_PATH, models["yolo"], models["hrnet"])
    joint_coordinates = get_pose3D(VIDEO_PATH, models["poseformer"], POSEFORMER_ARGS, keypoints)

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

        cv2.imshow('Matplotlib 3D Pose via OpenCV', img_bgr)
        cv2.waitKey(0)

    plt.clf()
    plt.close(fig)


# if __name__ == "__main__":
#     main()


# things the backend needs to do:
# 1. compute joint coordinates for each frame
# 2. compute loss score based on the coordinates and the selected reference video
# 3. Send down the reference video with joints overlaid
# 4. Generate a video of the user's inputted video with joints overlaid
# 5. Generate a video of the user's inputted video with skeleton in 3d space.
