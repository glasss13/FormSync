from gradio_client import Client, handle_file
import numpy as np
import cv2


image = np.load("steph_actual.npy")
cv2.imwrite("steph_actual_image.jpg", image)

client = Client("not-lain/background-removal")
result = client.predict(
		image=handle_file("steph_actual_image.jpg"),
		api_name="/image"
)

path = result[0]
cap = cv2.VideoCapture(path)
while cap.isOpened():
    
    ret, frame = cap.read()
    cv2.imshow("", frame)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if not ret:
        break  # Stop when video ends
    np.save("steph_background_removed.npy", frame)

