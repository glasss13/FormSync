import numpy as np
import cv2
from gradio_client import Client, handle_file
import numpy as np


frame_num = 17
client = Client("depth-anything/Video-Depth-Anything")
result = client.predict(
		input_video={"video":handle_file('input_vid.mp4')},
		max_len=500,
		target_fps=15,
		max_res=1280,
		api_name="/infer_video_depth"
)


depth_path = result[1]["video"]
actual_path = result[0]["video"]

cap = cv2.VideoCapture(depth_path)

depth_frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop when video ends
    depth_frames.append(frame)  # Append the frame as a NumPy array

cap.release()

depth_frames = np.array(depth_frames)  # Convert list to NumPy array
grey_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in depth_frames]


np.save("frame_depth.npy", grey_frames[frame_num])



cap = cv2.VideoCapture(actual_path)
actual_frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop when video ends
    actual_frames.append(frame)  # Append the frame as a NumPy array

cap.release()

actual_frames = np.array(actual_frames)  # Convert list to NumPy array
cv2.imwrite("frame_actual_image.jpg",actual_frames[frame_num])
np.save("frame.npy", actual_frames[frame_num])