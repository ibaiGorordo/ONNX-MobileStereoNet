import os
from dataclasses import dataclass
import numpy as np
import cv2
from google_drive_downloader import GoogleDriveDownloader as gdd

@dataclass
class CameraConfig:
    baseline: float
    f: float

def draw_disparity(disparity_map):

	min_val = np.min(disparity_map)
	max_val = np.max(disparity_map)
	norm_disparity_map = (255*((disparity_map-min_val)/(max_val - min_val))).astype(np.uint8)
	return cv2.applyColorMap(cv2.convertScaleAbs(norm_disparity_map,1), cv2.COLORMAP_JET)

def draw_depth(depth_map, max_dist):
	
	norm_depth_map = 255*(1-depth_map/max_dist)
	norm_depth_map[norm_depth_map < 0] =0
	norm_depth_map[depth_map == 0] =0

	return cv2.applyColorMap(cv2.convertScaleAbs(norm_depth_map,1), cv2.COLORMAP_JET)

def download_gdrive_tar_model(gdrive_id, model_path):

    if not os.path.exists(model_path):
        gdd.download_file_from_google_drive(file_id=gdrive_id,
                                    dest_path='./tmp/tmp.tar.gz')
        tar = tarfile.open("tmp/tmp.tar.gz", "r:gz")
        tar.extractall(path="tmp/")
        tar.close()

        shutil.move("tmp/saved_model_512x512/model_float32_opt.onnx", model_path)
        shutil.rmtree("tmp/")

def download_gdrive_file_model(model_path, gdrive_id):
    if not os.path.exists(model_path):
        gdd.download_file_from_google_drive(file_id=gdrive_id,
                                    dest_path=model_path)



