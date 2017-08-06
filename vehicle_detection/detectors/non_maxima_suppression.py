# import the necessary packages
import numpy as np

def non_maxima_suppression(bounding_boxes, confidences, overlap_thresh):
    # if there are no boxes, return an empty list
    if len(bounding_boxes) == 0:
        return []

    # Initialize the list of selected boxes:
    selected = []

    # Format datatype:
    bounding_boxes = bounding_boxes.astype(np.float)

    # Parse the coordinates of bounding boxes:
    top = bounding_boxes[:, 0]
    bottom = bounding_boxes[:, 1]
    left = bounding_boxes[:, 2]
    right = bounding_boxes[:, 3]

    # Compute the area of the bounding boxes
    area = (bottom - top + 1) * (right - left + 1)
    # Sort the bounding boxes by corresponding confidences
    rank = np.argsort(confidences)

    # Loop until the rank list is empty:
    while len(rank) > 0:
        # Select the most confident:
        most_confident_index = len(rank) - 1
        selected_index = rank[most_confident_index]
        selected.append(selected_index)

        # Calculate overlap ratio:
        top_intersect = np.maximum(top[selected_index], top[rank[:most_confident_index]])
        bottom_intersect = np.minimum(bottom[selected_index], bottom[rank[:most_confident_index]])
        left_intersect = np.maximum(left[selected_index], left[rank[:most_confident_index]])
        right_intersect = np.minimum(right[selected_index], right[rank[:most_confident_index]])

        h_intersect = np.maximum(0, bottom_intersect - top_intersect + 1)
        w_intersect = np.maximum(0, right_intersect - left_intersect + 1)

        overlap_ratio = (h_intersect * w_intersect) / area[rank[:most_confident_index]]

        # Delete all indexes from the rank list that have overlap greater than the
        # provided overlap threshold
        rank = np.delete(
            rank,
            np.concatenate(
                ([most_confident_index], np.where(overlap_ratio > overlap_thresh)[0])
            )
        )

	# return only the bounding boxes that were picked
    return (
        bounding_boxes[selected].astype(np.int),
        confidences[selected]
    )
