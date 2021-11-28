import cv2
import pafy
import numpy as np
import glob

from mobilestereonet import MobileStereoNet, CameraConfig
from mobilestereonet.utils import draw_depth

# Initialize video
# cap = cv2.VideoCapture("video.mp4")

videoUrl = 'https://youtu.be/Yui48w71SG0'
videoPafy = pafy.new(videoUrl)
print(videoPafy.streams)
cap = cv2.VideoCapture(videoPafy.getbestvideo().url)

model_path = "models/model_528_240_float32.onnx"

# Store baseline (m) and focal length (pixel)
input_width = 528
camera_config = CameraConfig(0.1, 0.5*input_width) # 90 deg. FOV
max_distance = 5

# Initialize model
mobile_depth_estimator = MobileStereoNet(model_path, camera_config)

cv2.namedWindow("Estimated depth", cv2.WINDOW_NORMAL)	
while cap.isOpened():

	try:
		# Read frame from the video
		ret, frame = cap.read()
		if not ret:	
			break
	except:
		continue

	# Extract the left and right images
	left_img  = frame[:,:frame.shape[1]//3]
	right_img = frame[:,frame.shape[1]//3:frame.shape[1]*2//3]
	color_real_depth = frame[:,frame.shape[1]*2//3:]

	# Estimate the depth
	disparity_map = mobile_depth_estimator(left_img, right_img)
	depth_map = mobile_depth_estimator.get_depth()

	color_depth = draw_depth(depth_map, max_distance)
	color_depth = cv2.resize(color_depth, (left_img.shape[1],left_img.shape[0]))
	cobined_image = np.hstack((left_img,color_real_depth, color_depth))

	cv2.imshow("Estimated depth", cobined_image)

	# Press key q to stop
	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
