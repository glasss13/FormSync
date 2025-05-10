from collections import defaultdict
import io
import json
import os
import tempfile
from types import SimpleNamespace
from uuid import uuid4
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

from flask import Flask, request, abort, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

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
    "./reference_videos/bbal.mp4",
    "./reference_videos/squat.mp4",
    "./reference_videos/klay.mp4"
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

ID_TO_ANGLE = {
    0: (1, 0, 4), # hip breadth
    1: (1, 0, 7), # right hip flexion
    2: (4, 0, 7), # Left hip flexion
    3: (0, 1, 2), # Right hip joint
    4: (1, 2, 3), # Right knee flexion
    5: (0, 4, 5), # Left hip joint
    6: (4, 5, 6), # Left knee flexion
    7: (0, 7, 8), # Spinal extension
    8: (7, 8, 9), # Lower cervical
    9: (7, 8, 11), # Left neck-to-shoulder
    10: (7, 8, 14), # Right neck-to-shoulder
    11: (9, 8, 11), # Upper left shoulder
    12: (9, 8, 14), # Upper right shoulder
    13: (11, 8, 14), # Shoulder spread
    14: (8, 9, 10), # Head tilt angle
    15: (8, 11, 12), # Left shoulder abduction
    16: (11, 12, 13), # Left elbow flexion
    17: (8, 14, 15), # Right shoulder abduction
    18: (14, 15, 16), # Right elbow flexion
}

ANGLE_TO_ID = {angle: id for (id, angle) in ID_TO_ANGLE.items()}

def create_pose_comparison_video(reference_joint_coordinates, joint_coordinates, path, video_filename="pose_comparison.mp4", fps=10):
    fig = plt.figure(figsize=(19.2, 5.4))
    gs = gridspec.GridSpec(1, 2)
    gs.update(wspace=0.05, hspace=0.05)

    ax1 = fig.add_subplot(gs[0], projection='3d', label="Reference video")
    ax2 = fig.add_subplot(gs[1], projection='3d', label="Your video")

    # Get the figure's dimensions in pixels to initialize VideoWriter
    # We draw the canvas once to ensure its size is set.
    fig.canvas.draw()
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png', dpi=100) # dpi can be adjusted
    img_buf.seek(0)
    img = plt.imread(img_buf)
    height, width, _ = img.shape
    img_buf.close()

    # Define the codec and create VideoWriter object
    # Common codecs: 'mp4v' for .mp4, 'XVID' for .avi
    # You might need to install codecs on your system (e.g., ffmpeg)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

    for step, (ref_idx, user_idx) in enumerate(path):
        ref_pose = reference_joint_coordinates[ref_idx]
        user_pose = joint_coordinates[user_idx]

        ax1.clear()
        ax2.clear()

        show3Dpose(ref_pose, ax1)
        show3Dpose(user_pose, ax2)

        ax1.text2D(0.5, 0.95, "Reference video", transform=ax1.transAxes, ha='center', va='top', fontsize=12, color='black')
        ax2.text2D(0.5, 0.95, "Your video", transform=ax2.transAxes, ha='center', va='top', fontsize=12, color='black')

        # Save the current figure to an in-memory buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100) # Ensure dpi matches initialization
        buf.seek(0)

        # Read the image from the buffer using OpenCV
        img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        buf.close()

        # OpenCV uses BGR by default, Matplotlib uses RGB. Convert if necessary.
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        out.write(frame_bgr)
        print(f"Processed frame {step+1}/{len(path)}")


    # Release everything when job is finished
    out.release()
    plt.clf() # Clear the current figure
    plt.close(fig) # Close the figure window

    print(f"Video '{video_filename}' created successfully.")

    return video_filename

def normalize_frame(joint_coords, reference_joint_idx=0):
    return joint_coords - joint_coords[reference_joint_idx]

def frame_distance(joint_coords_a, joint_coords_b, weights_dict = None):
    angles_a = extract_angles_from_frame(joint_coords_a)
    angles_b = extract_angles_from_frame(joint_coords_b)

    if weights_dict is None:
        weights_dict = {}

    for key in ANGLE_TO_ID.keys():
        if key not in weights_dict:
            weights_dict[key] = 1

    distance = 0
    distance_dict = {}

    for key in angles_a:
        angle_a = angles_a[key]
        angle_b = angles_b[key]
        diff = abs(angle_a - angle_b)
        diff = diff % 360
        if diff > 180:
            diff = 360 - diff
        distance += weights_dict[key] * diff
        distance_dict[key] = diff
    
    return distance, distance_dict

def align_videos(reference_video, user_video, weights_dict = None):
    max_frames = max(len(reference_video), len(user_video))
    _, warping_path = fastdtw(reference_video, user_video, radius=max_frames, dist=lambda x,y: frame_distance(x,y, weights_dict)[0])
    return warping_path

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

    if "weights" not in request.form:
        print("no weights provided")
        abort(400)

    try:
        weights_data = json.loads(request.form["weights"])
    except:
        print("Invalid weights JSON")
        abort(400)

    weights_dict = {ID_TO_ANGLE[int(id)]: weight for id, weight in weights_data.items()}
    for key in ANGLE_TO_ID.keys():
        if key not in weights_dict:
            weights_dict[key] = 0
    
    reference_joint_coordinates = REFERENCE_VIDEO_JOINTS[reference_id]

    print("aligning videos...")
    path = align_videos(reference_joint_coordinates, joint_coordinates)


    video_name = str(uuid4()) + ".mp4"
    create_pose_comparison_video(reference_joint_coordinates, joint_coordinates, path, f"./generated_videos/{video_name}")

    total_loss = 0
    joint_loss = defaultdict(lambda: 0)
    for step, (ref_idx, user_idx) in enumerate(path):
        ref_pose = reference_joint_coordinates[ref_idx]
        user_pose = joint_coordinates[user_idx]

        loss, dist_dict = frame_distance(ref_pose, user_pose, weights_dict)
        for k, v in dist_dict.items():
            joint_loss[k] += v

        print("distance: ", loss)
        total_loss += loss


    print("joint_loss:", joint_loss)

    for k, v in joint_loss.items():
        joint_loss[k] /= len(path)
        joint_loss[k] *= weights_dict[k]

    weights_sum = 0
    for v in weights_dict.values():
        weights_sum += v

    total_loss *= len(ID_TO_ANGLE) / weights_sum

    max_avg_loss = 180 * 19
    avg_loss = total_loss / len(path)


    response = {
        "avg_loss": float(avg_loss),
        "joint_loss":  {ANGLE_TO_ID[k]: float(v) for k,v in joint_loss.items()},
        "video_name": video_name
    }

    return jsonify(response), 200


@app.route("/videos/<path:filename>")
def serve_video(filename):
    return send_from_directory("generated_videos", filename)

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
