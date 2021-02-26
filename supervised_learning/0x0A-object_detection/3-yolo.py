#!/usr/bin/env python3
"""
Uses the Yolo v3 algorithm to perform object detection
"""


import tensorflow.keras as K
import numpy as np


class Yolo:
    """
    Uses the Yolo v3 algorithm to perform object detection
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor
        """
        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """
        Sigmoid function
        """
        return (1 / (1 + np.exp(-x)))

    def process_outputs(self, outputs, image_size):
        """
        Returns a tuple of (boxes, box_confidences, box_class_probs)
        """
        # Image’s original size
        image_height, image_width = image_size[0], image_size[1]

        boxes = []
        box_confidences = []
        box_class_probs = []

        for output in outputs:
            # list with processed boundary boxes for each output
            boxes.append(output[..., 0:4])
            # list with box confidences for each output
            box_confidences.append(self.sigmoid(output[..., 4, np.newaxis]))
            # list with box's class probabilities for each output
            box_class_probs.append(self.sigmoid(output[..., 5:]))

        for count, box in enumerate(boxes):
            grid_height, grid_width, anchor_boxes, _ = box.shape

            c = np.zeros((grid_height, grid_width, anchor_boxes), dtype=int)

            idx_y = np.arange(grid_height)
            idx_y = idx_y.reshape(grid_height, 1, 1)
            idx_x = np.arange(grid_width)
            idx_x = idx_x.reshape(1, grid_width, 1)
            # cx, cy: cell's top left corner of the box
            cy = c + idx_y
            cx = c + idx_x

            # The network predicts 4 coordinates for each bounding box
            t_x = (box[..., 0])
            t_y = (box[..., 1])

            # normalize the above variables
            t_x_n = self.sigmoid(t_x)
            t_y_n = self.sigmoid(t_y)

            # width and height
            t_w = (box[..., 2])
            t_h = (box[..., 3])

            """
            If the cell is offset from the top left corner of the
            image by (cx, cy) and the bounding box prior has width and
            height pw, ph, then the predictions correspond to
            """

            # center
            bx = t_x_n + cx
            by = t_y_n + cy

            # normalization
            bx /= grid_width
            by /= grid_height

            # priors (anchors) width and height
            pw = self.anchors[count, :, 0]
            ph = self.anchors[count, :, 1]

            # scale to anchors box dimensions
            bw = pw * np.exp(t_w)
            bh = ph * np.exp(t_h)

            # normalize to model input size
            input_width = self.model.input.shape[1].value
            input_height = self.model.input.shape[2].value
            bw /= input_width
            bh /= input_height

            # Corners of bounding box
            x1 = bx - bw / 2
            y1 = by - bh / 2
            x2 = x1 + bw
            y2 = y1 + bh

            # scale to image original size
            box[..., 0] = x1 * image_width
            box[..., 1] = y1 * image_height
            box[..., 2] = x2 * image_width
            box[..., 3] = y2 * image_height

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Returns a tuple of (filtered_boxes, box_classes, box_scores)
        """
        box_scores_full = []
        for box_conf, box_class_prob in zip(box_confidences, box_class_probs):
            box_scores_full.append(box_conf * box_class_prob)

        # box_scores
        box_scores_list = [score.max(axis=3) for score in box_scores_full]
        box_scores_list = [score.reshape(-1) for score in box_scores_list]
        box_scores = np.concatenate(box_scores_list)
        index_to_delete = np.where(box_scores < self.class_t)
        box_scores = np.delete(box_scores, index_to_delete)

        # box_classes
        box_classes_list = [box.argmax(axis=3) for box in box_scores_full]
        box_classes_list = [box.reshape(-1) for box in box_classes_list]
        box_classes = np.concatenate(box_classes_list)
        box_classes = np.delete(box_classes, index_to_delete)

        # filtered_boxes
        boxes_list = [box.reshape(-1, 4) for box in boxes]
        boxes = np.concatenate(boxes_list, axis=0)
        filtered_boxes = np.delete(boxes, index_to_delete, axis=0)

        return filtered_boxes, box_classes, box_scores

    @staticmethod
    def iou(box1, box2):
        """
        Intersection over union
        (x1, y1, x2, y2)
        """
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        intersection = max(yi2 - yi1, 0) * max(xi2 - xi1, 0)

        box1_area = (box1[3] - box1[1]) * (box1[2] - box1[0])
        box2_area = (box2[3] - box2[1]) * (box2[2] - box2[0])
        union = box1_area + box2_area - intersection

        return intersection / union

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Returns a tuple of
        (box_predictions, predicted_box_classes, predicted_box_scores)
        """
        index = np.lexsort((-box_scores, box_classes))

        box_predictions = np.array([filtered_boxes[i] for i in index])
        predicted_box_classes = np.array([box_classes[i] for i in index])
        predicted_box_scores = np.array([box_scores[i] for i in index])

        _, class_counts = np.unique(predicted_box_classes, return_counts=True)

        i = 0
        count = 0

        for class_count in class_counts:
            while i < count + class_count:
                j = i + 1
                while j < count + class_count:
                    temp = self.iou(box_predictions[i],
                                    box_predictions[j])
                    if temp > self.nms_t:
                        box_predictions = np.delete(box_predictions, j,
                                                    axis=0)
                        predicted_box_scores = np.delete(predicted_box_scores,
                                                         j, axis=0)
                        predicted_box_classes = (np.delete
                                                 (predicted_box_classes,
                                                  j, axis=0))
                        class_count = class_count - 1
                    else:
                        j = j + 1
                i = i + 1
            count = count + class_count

        return box_predictions, predicted_box_classes, predicted_box_scores