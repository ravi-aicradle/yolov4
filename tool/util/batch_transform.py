import os
import cv2
import torch
import logging
import numpy as np

logging.basicConfig(level=os.getenv("LOG_LEVEL", logging.DEBUG))
LOGGER = logging.getLogger(__name__)


class Transform:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    @torch.no_grad()
    def preprocess(self, unprocessed_batch_input):
        unprocessed_batch_input = np.array(unprocessed_batch_input)
        processed_batch = [
            cv2.resize(cv2.cvtColor(x, cv2.COLOR_BGR2RGB), (self.height, self.width), interpolation=cv2.INTER_LINEAR) / 255
            for x in
            unprocessed_batch_input]
        processed_batch_transpose = np.array([np.transpose(x, (2, 0, 1)).astype(np.float32) for x in processed_batch])
        processed_batch = torch.from_numpy(processed_batch_transpose)
        processed_batch = processed_batch.squeeze(0)
        return processed_batch