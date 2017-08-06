## Set up session:
import argparse
# Configuration file:
from vehicle_detection.utils.conf import Conf
# IO utilities:
import random
import glob
import pickle
# Image processing:
import numpy as np
import cv2
from vehicle_detection.extractors import ReshapeTransformer,ColorHistogramTransformer, HOGTransformer, TemplateTransformer

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
        help="Input video name."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output video name template."
    )
    parser.add_argument(
        "-s", "--start",
        type=int,
        required=True,
        help="Start second."
    )
    parser.add_argument(
        "-e", "--end",
        type=int,
        required=True,
        help="End second."
    )
    args = vars(parser.parse_args())

    # Load config:
    conf = Conf(args["config"])

    # Load image:
    print(args["input"])
    print(args["output"])
    print(args["start"])
    print(args["end"])
