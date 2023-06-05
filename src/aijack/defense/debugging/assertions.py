import cv2
import numpy as np

from .utils import Hungarian


class MultiBoxAssertionError(AssertionError):
    """MultiBoxAssertionError class."""

    def __init__(self, message):
        super().__init__(message)


def nearly_contains(box_1, box_2, eps):
    return box_1[0] + eps < box_2[2] and box_2[2] + eps < box_1[0]


def assert_multibox(boxes, counter_threshold=2, eps=0):
    counter = [0 for i in range(len(boxes))]
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if i == j:
                continue
            if nearly_contains(boxes[i], boxes[j], eps):
                counter[i] += 1
                if counter[i] >= counter_threshold:
                    raise MultiBoxAssertionError(
                        f"Box {i} overlaps more than {counter_threshold} other boxes."
                    )


class FlickerException(Exception):
    """FlickerException class."""

    def __init__(self, message):
        super().__init__(message)


def psnr(img_1, img_2, data_range=255, eps=1e-8):
    if img_1.shape != img_2.shape:
        new_shape = (
            max(img_1.shape[0], img_2.shape[0]),
            (max(img_1.shape[1], img_2.shape[1])),
        )
        img_1 = cv2.resize(img_1, new_shape)
        img_2 = cv2.resize(img_2, new_shape)
    mse = np.mean((img_1.astype(float) - img_2.astype(float)) ** 2)
    return 10 * np.log10((data_range**2) / (mse + eps))


def get_simillar_boxes(
    frame_1,
    frame_2,
    boxes_1,
    boxes_2,
    similarity_threshold=35,
    data_range=255,
    eps=1e-8,
):
    n1 = len(boxes_1)
    n2 = len(boxes_2)
    sim_matrix = np.zeros((max(n1, n2), max(n1, n2))) + (1 / eps)
    for i in range(n1):
        for j in range(n2):
            sim_matrix[i][j] = 1 / (
                psnr(
                    frame_1[
                        int(boxes_1[i][1]) : int(boxes_1[i][3]) :,
                        int(boxes_1[i][0]) : int(boxes_1[i][2]),
                    ],
                    frame_2[
                        int(boxes_2[j][1]) : int(boxes_2[j][3]) :,
                        int(boxes_2[j][0]) : int(boxes_2[j][2]),
                    ],
                    data_range,
                    eps,
                )
                + eps
            )
    h = Hungarian()
    assignment = h.compute(sim_matrix)
    sim_thres_inv = 1 / similarity_threshold
    similar_boxes = []
    for ass in assignment:
        if sim_matrix[ass[0]][ass[1]] < sim_thres_inv:
            similar_boxes.append(boxes_2[ass[1]])
    return similar_boxes


def except_flicker(
    frames,
    boxes_of_frames,
    cur_frame_id=-1,
    window_size=10,
    similarity_threshold=35,
    data_range=255,
):
    cur_frame = frames[cur_frame_id]
    cur_boxes = boxes_of_frames[cur_frame_id]
    for i in range(
        cur_frame_id - 1, max(-len(frames), cur_frame_id - window_size) - 1, -1
    ):
        similar_boxes = get_simillar_boxes(
            cur_frame,
            frames[i],
            cur_boxes,
            boxes_of_frames[i],
            similarity_threshold,
            data_range,
        )
        if len(similar_boxes) == 0:
            for j in range(
                i - 1, max(-len(frames), cur_frame_id - window_size) - 1, -1
            ):
                overlapping_boxes = get_simillar_boxes(
                    cur_frame,
                    frames[j],
                    cur_boxes,
                    boxes_of_frames[j],
                    similarity_threshold,
                    data_range,
                )
                if len(overlapping_boxes) == 0:
                    continue
                else:
                    raise FlickerException(
                        f"Flickering detected between frame {cur_frame_id} and frame {j}"
                    )
        else:
            cur_frame = frames[i]
            cur_boxes = similar_boxes
