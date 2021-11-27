import cv2
import numpy as np
from imread_from_url import imread_from_url

from mobilestereonet import MobileStereoNet, CameraConfig
from mobilestereonet.utils import draw_disparity

if __name__ == '__main__':
	
	model_path = "models/model_528_240_float32.onnx"

	# Initialize model
	mobile_depth_estimator = MobileStereoNet(model_path)

	# Load images
	left_img = imread_from_url("https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im2.png")
	right_img = imread_from_url("https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im6.png")

	# Estimate the depth
	disparity_map = mobile_depth_estimator(left_img, right_img)

	color_disparity = draw_disparity(disparity_map)
	color_disparity = cv2.resize(color_disparity, (left_img.shape[1],left_img.shape[0]))

	combined_image = np.hstack((left_img, right_img, color_disparity))

	cv2.imwrite("out.jpg", combined_image)

	cv2.namedWindow("Estimated disparity", cv2.WINDOW_NORMAL)	
	cv2.imshow("Estimated disparity", combined_image)
	cv2.waitKey(0)

	cv2.destroyAllWindows()

