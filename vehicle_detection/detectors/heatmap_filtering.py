# Set up session:
import numpy as np
from scipy.ndimage.measurements import label

def heatmap_filtering(image, bounding_boxes, heat_thresh):
    # If there are no boxes, return an empty list
    if len(bounding_boxes) == 0:
        return []

    # Initialize heatmap:
    H, W, _ = image.shape
    heatmap = np.zeros((H, W), dtype=np.int)

    # Aggregate heat:
    for bounding_box in bounding_boxes:
        (top, bottom, left, right) = bounding_box
        heatmap[top:bottom, left:right] += 1

    # Filter:
    heatmap[heatmap <= heat_thresh] = 0

    # Label it:
    labelled, num_components = label(heatmap)

    # Identify external bounding boxes:
    external_bounding_boxes = []
    for component_id in range(1, num_components + 1):
        # Find pixels with each car_number label value
        nonzero = (labelled == component_id).nonzero()
        # Identify x and y values of those pixels
        nonzero_y, nonzero_x = np.array(nonzero[0]), np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        external_bounding_boxes.append(
            (
                np.min(nonzero_y),
                np.max(nonzero_y),
                np.min(nonzero_x),
                np.max(nonzero_x)
            )
        )

    return external_bounding_boxes
