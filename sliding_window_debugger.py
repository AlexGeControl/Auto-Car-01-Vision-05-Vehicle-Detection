## Set up session:
import argparse
import pickle

import numpy as np
import cv2

from vehicle_detection.utils.conf import Conf
from vehicle_detection.detectors.image_processing import resize, sliding_window

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Parse command-line arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to JSON configuration file."
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to input image."
    )
    args = vars(parser.parse_args())

    # Load config:
    conf = Conf(args["config"])

    # Load classifer:
    with open(conf.classifier_path, 'rb') as classifier_pkl:
        classifier = pickle.load(classifier_pkl)

    # Load image:
    image = cv2.imread(args["input"])
    for image_channel in cv2.split(
        cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    ):
        cv2.imshow("Input", image_channel)
        cv2.waitKey(0)

    # Raw image dimensions:
    IMG_H, IMG_W, _ = image.shape

    for level, sliding_window_config in conf.sliding_window_pyramid:
        print("[{}]: ...".format(level))

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

        # Generate image patches using sliding window:
        canvas = image.copy()
        # ROI:
        cv2.rectangle(
            canvas,
            (img_left, img_top), (img_right, img_bottom),
            (255, 0, 0),
            8
        )

        bounding_boxes = []
        for ((top, left), patch) in sliding_window(
            ROI,
            (WIN_W, WIN_H),
            (W_STEP, H_STEP)
        ):
            patch= cv2.resize(
                patch,
                tuple(conf.hog_window_size),
                interpolation = cv2.INTER_AREA
            )

            patch = patch.reshape(conf.shape_serialized)
            if getattr(classifier, "predict_proba", None) is None:
                confidence = classifier.predict(patch)[0]
            else:
                confidence = classifier.predict_proba(patch)[:, 1]

            bounding_box = (
                img_top + int(SCALE * top),
                img_top + int(SCALE * (top + WIN_H)),
                img_left + int(SCALE * left),
                img_left + int(SCALE * (left + WIN_W))
            )
            bounding_boxes.append(bounding_box)

            if confidence > conf.confidence_thresh:
                patch_top, patch_bottom, patch_left, patch_right = bounding_box
                # Patch:
                cv2.rectangle(
                    canvas,
                    (patch_left, patch_top), (patch_right, patch_bottom),
                    (0, 255, 0),
                    6
                )

        cv2.destroyAllWindows()
        cv2.imshow("{}".format(level), canvas)
        cv2.imwrite(
            "sliding-window-pyramid-structure-demo-{}.png".format(level),
            canvas
        )
        cv2.waitKey(0)
