import sys
import os
import cv2
import time
import math
import torch
import numpy as np
import logging
from torch.autograd import Variable

import itertools
import struct  # get_image_size
import imghdr  # get_image_size

from tool import utils
from tool.utils import PostProcessing

logging.basicConfig(level=os.getenv("LOG_LEVEL", logging.DEBUG))
LOGGER = logging.getLogger(__name__)


def bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = torch.min(boxes1[0], boxes2[0])
        Mx = torch.max(boxes1[2], boxes2[2])
        my = torch.min(boxes1[1], boxes2[1])
        My = torch.max(boxes1[3], boxes2[3])
        w1 = boxes1[2] - boxes1[0]
        h1 = boxes1[3] - boxes1[1]
        w2 = boxes2[2] - boxes2[0]
        h2 = boxes2[3] - boxes2[1]
    else:
        mx = torch.min(boxes1[0] - boxes1[2] / 2.0, boxes2[0] - boxes2[2] / 2.0)
        Mx = torch.max(boxes1[0] + boxes1[2] / 2.0, boxes2[0] + boxes2[2] / 2.0)
        my = torch.min(boxes1[1] - boxes1[3] / 2.0, boxes2[1] - boxes2[3] / 2.0)
        My = torch.max(boxes1[1] + boxes1[3] / 2.0, boxes2[1] + boxes2[3] / 2.0)
        w1 = boxes1[2]
        h1 = boxes1[3]
        w2 = boxes2[2]
        h2 = boxes2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea / uarea


def get_region_boxes(boxes_and_confs):
    # print('Getting boxes from boxes and confs ...')

    boxes_list = []
    confs_list = []

    for item in boxes_and_confs:
        boxes_list.append(item[0])
        confs_list.append(item[1])

    # boxes: [batch, num1 + num2 + num3, 1, 4]
    # confs: [batch, num1 + num2 + num3, num_classes]
    boxes = torch.cat(boxes_list, dim=1)
    confs = torch.cat(confs_list, dim=1)

    return [boxes, confs]


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)


def do_detect(model, img, conf_thresh, nms_thresh, use_cuda=1):
    model.eval()
    t0 = time.time()

    if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    elif type(img) == np.ndarray and len(img.shape) == 4:
        img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
    else:
        print("unknow image type")
        exit(-1)

    if use_cuda:
        img = img.cuda()
    img = torch.autograd.Variable(img)

    t1 = time.time()

    output = model(img)

    t2 = time.time()

    print('-----------------------------------')
    print('           Preprocess : %f' % (t1 - t0))
    print('      Model Inference : %f' % (t2 - t1))
    print('-----------------------------------')

    return utils.post_processing(img, conf_thresh, nms_thresh, output)


class Transform:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def preprocess(self, unprocessed_batch_input):
        unprocessed_batch_input = np.array(unprocessed_batch_input)
        LOGGER.debug("Processing Image Batch of shape : {}".format(unprocessed_batch_input.shape))
        processed_batch = [
            cv2.resize(cv2.cvtColor(x, cv2.COLOR_BGR2RGB), (self.height, self.width), interpolation=cv2.INTER_LINEAR) / 255
            for x in
            unprocessed_batch_input]
        processed_batch_transpose = np.array([np.transpose(x, (2, 0, 1)).astype(np.float32) for x in processed_batch])
        processed_batch = torch.from_numpy(processed_batch_transpose)
        return processed_batch


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
        t0 = time.time()
        transformed_batch_input = self.transform.preprocess(batch_input)
        transformed_batch_input = transformed_batch_input.to(self.device)
        t1 = time.time()
        output = model(transformed_batch_input)
        t2 = time.time()
        """print('-----------------------------------')
        print('           Preprocess : %f' % (t1 - t0))
        print('      Model Inference : %f' % (t2 - t1))
        print('-----------------------------------')"""
        boxes_out = utils.post_processing(None, self.conf_thresh, self.nms_thresh, output)
        return boxes_out
