from collections import OrderedDict
import copy
from types import SimpleNamespace

import torch
import cv2
import numpy as np
    
from sort import Sort
from utils import PreProcess, get_final_preds, h36m_coco_format, revise_kpts
from yolo.darknet import Darknet
from yolo.human_detector import yolo_human_det

from hrnet import pose_hrnet
from hrnet.config import _C as hrnet_config, update_config

YOLO_CFG: str = "./yolo/yolov3.cfg"
YOLO_WEIGHT_PATH: str = "./yolo/yolov3.weights"


if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def load_YOLO_model(inp_dim=416):
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    model = Darknet(YOLO_CFG)
    model.load_weights(YOLO_WEIGHT_PATH)

    model.net_info["height"] = inp_dim

    model.to(DEVICE)
    model.eval()

    return model

def load_hrnet_model(config):
    model = pose_hrnet.get_pose_net(config, is_train=False)

    state_dict = torch.load(config.OUTPUT_DIR, map_location=DEVICE)
    new_state_dict = OrderedDict()  
    for k, v in state_dict.items():
        name = k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    model.eval()
    model.to(DEVICE)

    return model


def get_pose2d(video_path: str, human_model, pose_model):
    capture = cv2.VideoCapture(video_path)

    kpts, scores = gen_video_kpts(capture, human_model, pose_model)
    kpts, scores, valid_frames = h36m_coco_format(kpts, scores)

    kpts = revise_kpts(kpts, scores, valid_frames)
    return kpts



def debug_human_detection(frame, bboxs, scores):
    if bboxs is not None and len(bboxs) > 0:
        for bbox, score in zip(bboxs, scores):
            x1, y1, x2, y2 = map(int, bbox)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label  =f"Human: {score[0]:.2f}"
            cv2.putText(frame, label, (x1, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        print("No humans detected")
    cv2.imshow("human", frame)
    cv2.waitKey(0)



def gen_video_kpts(video_capture: cv2.VideoCapture, human_model, pose_model, det_dim=416, num_person=1, gen_output=False):
    
    # Loading detector and pose model, initialize sort for track
    # human_model = load_YOLO_model(inp_dim=det_dim)

    # update_config(hrnet_config, hrnet_args)
    # pose_model = load_hrnet_model(hrnet_config)

    people_sort = Sort(min_hits=0)

    video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    kpts_result = []
    scores_result = []
    for frame_idx in range(video_length):
        ret, frame = video_capture.read()

        if not ret:
            continue

        bboxs, scores = yolo_human_det(frame, human_model, reso=det_dim, device=DEVICE)
        # debug_human_detection(frame, bboxs, scores)

        if bboxs is None or not bboxs.any():
            print('No person detected!')
            bboxs = bboxs_pre
            scores = scores_pre
        else:
            bboxs_pre = copy.deepcopy(bboxs) 
            scores_pre = copy.deepcopy(scores) 

        # Using Sort to track people
        people_track = people_sort.update(bboxs)

        # Track the first two people in the video and remove the ID
        if people_track.shape[0] == 1:
            people_track_ = people_track[-1, :-1].reshape(1, 4)
        elif people_track.shape[0] >= 2:
            people_track_ = people_track[-num_person:, :-1].reshape(num_person, 4)
            people_track_ = people_track_[::-1]
        else:
            continue

        track_bboxs = []
        for bbox in people_track_:
            bbox = [round(i, 2) for i in list(bbox)]
            track_bboxs.append(bbox)

        with torch.no_grad():
            # bbox is coordinate location
            inputs, origin_img, center, scale = PreProcess(frame, track_bboxs, hrnet_config, num_person)

            inputs = inputs[:, [2, 1, 0]]
            inputs = inputs.to(DEVICE)

            output = pose_model(inputs)

            # compute coordinate
            preds, maxvals = get_final_preds(hrnet_config, output.clone().cpu().numpy(), np.asarray(center), np.asarray(scale))

        kpts = np.zeros((num_person, 17, 2), dtype=np.float32)
        scores = np.zeros((num_person, 17), dtype=np.float32)
        for i, kpt in enumerate(preds):
            kpts[i] = kpt

        for i, score in enumerate(maxvals):
            scores[i] = score.squeeze()

        kpts_result.append(kpts)
        scores_result.append(scores)

    keypoints = np.array(kpts_result)
    scores = np.array(scores_result)

    keypoints = keypoints.transpose(1, 0, 2, 3)  # (T, M, N, 2) --> (M, T, N, 2)
    scores = scores.transpose(1, 0, 2)  # (T, M, N) --> (M, T, N)

    return keypoints, scores
