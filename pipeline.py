import os
from typing import Tuple
import cv2
import numpy as np

import detectron
import ocr_eval


class Pipeline:
    def __init__(self, vid_path: str, img_size: Tuple[int, int], seg_model_dir: str, frame_dir: str = "",
                 masked_dir: str = ""):
        self.vid_path = vid_path
        self.img_size = img_size

        self.frame_dir = frame_dir
        self.masked_dir = masked_dir
        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)
        if not os.path.exists(masked_dir):
            os.makedirs(masked_dir)

        self.seg_model = detectron.get_predictor(seg_model_dir)

        self.performance_labels = []

    def vid_to_frames(self, start_frame: int = 0, skip_frames: int = 9):
        vid_cap = cv2.VideoCapture(self.vid_path)

        for _ in range(start_frame):
            vid_cap.read()

        max_images = float("inf")
        count = 0

        success, image = vid_cap.read()
        while success and count < max_images:
            img_filename = str(count).rjust(5, "0") + ".jpg"
            cv2.resize(image, self.img_size)
            cv2.imwrite(self.frame_dir + img_filename, image)

            # Skip frames to reduce frame rate
            for _ in range(skip_frames):
                vid_cap.read()
        
            # Read new frame
            success, image = vid_cap.read()
            print('Read a new frame:', img_filename)
            count += 1

    def mask_frames(self):
        for filename in os.listdir(self.frame_dir):
            if filename.endswith('.jpg'):
                img = cv2.imread(os.path.join(self.frame_dir, filename))
                if img is not None:
                    mask_data = self.seg_model(img)['instances'].to('cpu').get_fields()
                    centers = np.array(mask_data['pred_boxes'].get_centers())
                    indices = np.apply_along_axis(lambda row: row[0], 1, centers).argsort()
                    for i, mask in enumerate(mask_data['pred_masks'][indices]):
                        mask = np.repeat(mask.numpy().astype('int')[:, :, np.newaxis], 3, axis=2)
                        masked_img = img * mask
                        cv2.imwrite(os.path.join(self.masked_dir, filename[:-4] + f'.{i}.jpg'), masked_img)

    def compute_performance_labels(self):
        for filename in os.listdir(self.masked_dir):
            if filename.endswith('.jpg'):
                img = cv2.imread(os.path.join(self.masked_dir, filename))
                if img is not None:
                    text = ocr_eval.perform_ocr(img)
