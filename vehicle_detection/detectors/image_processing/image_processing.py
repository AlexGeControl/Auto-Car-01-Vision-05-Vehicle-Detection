# Set up session:
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
import cv2

def resize(image, output_height=None, output_width=None, interpolation=cv2.INTER_AREA):
    """ Resize the image to desired size. Optimized for decimation.
    """
    input_height, input_width = image.shape[:2]
    output_size = None

    if (output_height is None) or (output_width is None):
        # The output size is not specified:
        if (output_height is None) and (output_width is None):
            return image
        elif output_height is None:
            ratio = output_width / input_width
            output_height = ratio * input_height
        elif output_width is None:
            ratio = output_height / input_height
            output_width = ratio * input_width

    # Output size specification--x-coord first then y-coord
    output_size = (int(output_width), int(output_height))

    return cv2.resize(image, output_size, interpolation=interpolation)

def pyramid(image, scales, min_size=(32, 32)):
    """ Image pyramid generator
    """
    # Parse image shape:
    image_H, image_W, _ = image.shape
    # Min image shape:
    min_H, min_W = min_size

    yield (image, 1.0)

    for scale in scales:
        scaled_H, scaled_W = image_H / scale, image_W / scale

        if scaled_H < min_H or scaled_W < min_W:
            break

        image_scaled = resize(image, output_height=scaled_H, output_width=scaled_W)

        yield (image_scaled, scale)

def sliding_window(image, window_size, step_size):
    """ Sliding window generator
    """
    max_H, max_W = image.shape[:2]

    # Sliding window config:
    win_W, win_H = window_size
    W_step_size, H_step_size = step_size

    # Do not use max size correction here for image padding case:
    for top in range(0, max_H - win_H + 1, H_step_size):
        for left in range(0, max_W - win_W + 1, W_step_size):
            bottom, right = top + win_H, left + win_W
            yield ((top, left), image[top:bottom, left:right])

def canonicalize(image, bounding_box, padding, canonical_size):
    """ Canonicalize the image patch for object detection
    """
    # Parse bounding box:
    (top, bottom, left, right) = bounding_box
    # Padding:
    (top_padded, left_padded) = (max(top - padding, 0), max(left - padding, 0))

    # Canonicalize:
    ROI = cv2.resize(
        image[
            top_padded:(bottom + padding), left_padded:(right + padding)
        ],
        canonical_size,
        interpolation=cv2.INTER_AREA
    )

    return ROI

def extract_random_patches(image, patch_size, max_patches):
    """ Extract random patches from given image
    """
    return extract_patches_2d(image, patch_size, max_patches=max_patches)

def auto_canny(image, sigma = 0.33):
    """ Heuristically optimal canny edge detector

        1. Remove noise using Gaussian smoothing;
        2. Suppress non-local maximum pixel to keep only thin edges;
        3. Use hysteresis thresholding to keep only strong edges.
    """
    # compute the median of the single channel pixel intensities
    median = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    edged = cv2.Canny(image, lower, upper)

    return edged
