import os
import cv2
import torch
import logging
import numpy as np

logging.basicConfig(level=os.getenv("LOG_LEVEL", logging.DEBUG))
LOGGER = logging.getLogger(__name__)


class BaseTransform:
    def __init__(self, max_size):
        self.max_size = max_size

    def transform(self, unprocessed_batch):
        if len(unprocessed_batch.shape) == 3:
            unprocessed_batch = cv2.cvtColor(unprocessed_batch, cv2.COLOR_BGR2RGB)
            processed_batch = cv2.resize(unprocessed_batch, (416, 416), interpolation=cv2.INTER_LINEAR) / 255
            processed_batch_tras = np.transpose(processed_batch, (2, 0, 1)).astype(np.float32)
            processed_batch = torch.from_numpy(processed_batch_tras)
        else:
            # not going to use this as different workers will pre-process images sepearted and we finally stack them together as a batch in the worker
            LOGGER.debug("Processing Image Batch of shape : {}".format(unprocessed_batch.shape))
            processed_batch = [cv2.resize(cv2.cvtColor(x, cv2.COLOR_BGR2RGB), (416, 416), interpolation=cv2.INTER_LINEAR) / 255 for x in
                               unprocessed_batch]
            processed_batch_transpose = [np.transpose(x, (2, 0, 1)).astype(np.float32) for x in processed_batch]
            processed_batch = torch.from_numpy(processed_batch_transpose)

        return processed_batch