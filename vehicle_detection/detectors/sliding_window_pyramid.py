## Set up session:
import pickle

import numpy as np
import cv2

from .image_processing import resize, sliding_window

class SlidingWindowPyramidDetector:
    def __init__(self, conf):
        # Load classifer:
        with open(conf.classifier_path, 'rb') as classifier_pickle:
            self.classifier = pickle.load(classifier_pickle)
        # Save config:
        self.conf = conf

    def detect(self, image):
        """ Detect vehicles in given image
        """
        # Initialize:
        patches = []
        bounding_boxes = []

        # Raw image dimensions:
        IMG_H, IMG_W, _ = image.shape

        # Generate image pyramid:
        for level, sliding_window_config in self.conf.sliding_window_pyramid:
            # Extract ROI:
            img_top_percent, img_bottom_percent, img_left_percent, img_right_percent = sliding_window_config["ROI"]
            img_top, img_bottom, img_left, img_right = (
                int(   img_top_percent*IMG_H),
                int(img_bottom_percent*IMG_H),
                int(  img_left_percent*IMG_W),
                int( img_right_percent*IMG_W)
            )
            ROI = image[img_top:img_bottom, img_left:img_right]

            # Sliding window size:
            WIN_W, WIN_H = sliding_window_config["window_size"]
            W_STEP, H_STEP = sliding_window_config["window_step"]

            # Resize:
            ROI_H, ROI_W, _ = ROI.shape
            SCALE = sliding_window_config["scale"]
            ROI_H, ROI_W = ROI_H / SCALE, ROI_W / SCALE

            if ROI_W < WIN_W or ROI_H < WIN_H:
                continue

            ROI = resize(ROI, output_height=ROI_H, output_width=ROI_W)

            # Generate sliding window:
            for ((top, left), patch) in sliding_window(
                ROI,
                (WIN_W, WIN_H),
                (W_STEP, H_STEP)
            ):
                # Save patch:
                patches.append(
                    cv2.resize(
                        patch,
                        tuple(self.conf.hog_window_size),
                        interpolation = cv2.INTER_AREA
                    )
                )

                # Save bounding box:
                bounding_boxes.append(
                    (
                        img_top + int(SCALE * top),
                        img_top + int(SCALE * (top + WIN_H)),
                        img_left + int(SCALE * left),
                        img_left + int(SCALE * (left + WIN_W))
                    )
                )

        # Classify:
        patches = np.array(patches).reshape(self.conf.shape_serialized)
        if getattr(self.classifier, "predict_proba", None) is None:
            confidences = self.classifier.predict(patches)
        else:
            confidences = self.classifier.predict_proba(patches)[:, 1]

        # Filter by confidence:
        bounding_boxes = np.array(bounding_boxes)[np.where(confidences > self.conf.confidence_thresh)[0]]

        # Finally:
        return bounding_boxes
