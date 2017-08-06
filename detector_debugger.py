## Set up session:
import argparse
import pickle

import numpy as np
import cv2

from vehicle_detection.utils.conf import Conf
from vehicle_detection.detectors import SlidingWindowPyramidDetector
from vehicle_detection.detectors import heatmap_filtering

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

    # Load image:
    image = cv2.imread(args["input"])

    # Initialize detector:
    detector = SlidingWindowPyramidDetector(
        conf
    )

    # Detect:
    bounding_boxes = detector.detect(
        image
    )

    # Heatmap filtering:
    bounding_boxes = heatmap_filtering(image, bounding_boxes, conf.heat_thresh)

    # Draw:
    canvas = image.copy()
    for bounding_box in bounding_boxes:
        (top, bottom, left, right) = bounding_box
        cv2.rectangle(
            canvas,
            (left, top), (right, bottom),
            (0, 255, 0),
            6
        )

    cv2.imshow("Detected", canvas)
    cv2.waitKey(0)
