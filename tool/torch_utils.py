import os
import numpy as np
import torch
import logging

logging.basicConfig(level=os.getenv("LOG_LEVEL", logging.DEBUG))
LOGGER = logging.getLogger(__name__)


class Yolov4Classifier:
    def __init__(self, post_processing, transform, confidence_threshold, nms_threshold, device, height, width):
        self.conf_thresh = confidence_threshold
        self.nms_thresh = nms_threshold
        self.device = device
        self.height = height
        self.width = width
        self.post_processing = post_processing
        self.transform = transform

    def predict_batch(self, model, batch_input):
        model.eval()
        transformed_batch_input = self.transform.preprocess(batch_input)
        transformed_batch_input = transformed_batch_input.to(self.device)
        prediction = model(transformed_batch_input)
        boxes_out = self.post_processing.post_processing(prediction)
        return boxes_out
