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
        return None

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Returns a tuple of (filtered_boxes, box_classes, box_scores)
        """
        return None

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Returns a tuple of
        (box_predictions, predicted_box_classes, predicted_box_scores)
        """
        return None

    @staticmethod
    def load_images(folder_path):
        """
        Returns a tuple of (images, image_paths)
        """
        return None
