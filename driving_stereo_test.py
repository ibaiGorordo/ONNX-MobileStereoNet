import cv2
import numpy as np
import glob
from mobilestereonet import MobileStereoNet, CameraConfig
from mobilestereonet.utils import draw_depth

# out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (5286//2,800//2))

# Get image list
left_images = glob.glob('DrivingStereo images/left/*.png')
left_images.sort()
right_images = glob.glob('DrivingStereo images/right/*.png')
right_images.sort()
depth_images = glob.glob('DrivingStereo images/depth/*.png')
depth_images.sort()

model_path = "models/model_528_240_float32.onnx"
input_width = 528
camera_config = CameraConfig(0.546, 2000/1920*input_width) # rough estimate from the original calibration
max_distance = 30

# Initialize model
mobile_depth_estimator = MobileStereoNet(model_path, camera_config)

cv2.namedWindow("Estimated depth", cv2.WINDOW_NORMAL)	
for left_path, right_path, depth_path in zip(left_images[:], right_images[:], depth_images[:]):	

	# Read frame from the video
	left_img = cv2.imread(left_path)
	right_img = cv2.imread(right_path)
	depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)/256

	# Estimate the depthq
	disparity_map = mobile_depth_estimator(left_img, right_img)
	depth_map = mobile_depth_estimator.get_depth()

	color_depth = draw_depth(depth_map, max_distance)
	color_real_depth = draw_depth(depth_img, max_distance)

	color_depth = cv2.resize(color_depth, (left_img.shape[1],left_img.shape[0]))
	combined_image = np.hstack((left_img, color_depth))
	combined_image = cv2.resize(combined_image, (combined_image.shape[1]//2,combined_image.shape[0]//2))

	# out.write(combined_image)
	cv2.imshow("Estimated depth", combined_image)

	# Press key q to stop
	if cv2.waitKey(1) == ord('q'):
		break

# out.release()
cv2.destroyAllWindows()