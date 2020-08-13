import torch
from torch import nn
import torch.nn.functional as F
from tool.torch_utils import *
from tool.torch_utils import Yolov4Classifier, Transform
from tool.yolo_layer import YoloLayer
from classifier.model import Mish, Upsample, Conv_Bn_Activation, ResBlock, DownSample1, DownSample2, DownSample3, \
    Neck, Yolov4Head, Yolov4
from argparse import ArgumentParser

from tool.utils import load_class_names, plot_boxes_cv2, PostProcessing
from tool.torch_utils import do_detect
from config.dot_style_configuration import DotStyleConfiguration

import sys
import cv2
from moviepy.editor import VideoFileClip

processed_images = list()
original_sizes = list()
original_images = list()


def main(args):
    global frame_count
    global model
    global use_cuda
    global parameter_config
    global yolov4

    parameter_config = DotStyleConfiguration(args.configuration_file_path)

    weightfile = parameter_config.model_config.classifier_path

    n_classes = parameter_config.model_config.num_classes
    input_video = args.input_video
    output_video = args.output_video

    model = Yolov4(yolov4conv137weight=None, n_classes=n_classes, inference=True)

    pretrained_dict = torch.load(weightfile, map_location=torch.device('cpu'))
    model.load_state_dict(pretrained_dict)

    use_cuda = False
    if use_cuda:
        model.cuda()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    frame_count = 0
    confidence_threshold = parameter_config.model_config.confidence_threshold
    nms_threshold = parameter_config.model_config.nms_threshold
    height, width = (parameter_config.model_config.image_height, parameter_config.model_config.image_width)

    post_processing = PostProcessing(confidence_threshold, nms_threshold)
    transform = Transform(height, width)
    yolov4 = Yolov4Classifier(post_processing, transform, confidence_threshold, nms_threshold, device, height, width)

    video = VideoFileClip(input_video)  # .subclip(50, 60)
    clip = video.fl_image(pipeline)
    clip.write_videofile(output_video, fps=args.fps)


def pipeline(original_image):
    global frame_count
    global model
    global use_cuda
    global parameter_config
    global yolov4
    height, width = (parameter_config.model_config.image_height, parameter_config.model_config.image_width)
    resized_image = cv2.resize(original_image, (width, height))
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    confidence_threshold = parameter_config.model_config.confidence_threshold
    nms_threshold = parameter_config.model_config.nms_threshold
    batch_size = parameter_config.batch_size
    device = parameter_config.model_config.device

    class_map = parameter_config.class_map
    class_names = 'config/coco_classes'
    class_names = load_class_names(class_names)
    # print('confidence_threshold, nms_threshold, device', confidence_threshold, nms_threshold, device)

    # if frame_count == 0 or frame_count % batch_size != 0:
    processed_images.append(resized_image)
    original_sizes.append([original_image.shape[0], original_image.shape[1]])
    original_images.append(original_image)
    if frame_count > 0 and frame_count % batch_size == 0:
        boxes = yolov4.predict_batch(model, processed_images)
        for idx, box in enumerate(boxes):
            savename = '/Users/ravikannan/Desktop/yolov4/'+str(frame_count-batch_size+idx)+'.jpg'
            img = plot_boxes_cv2(original_images[idx], box, savename, class_names)
        processed_images.clear()
        original_sizes.clear()
        original_images.clear()
    frame_count += 1

    return original_image


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_video', help="input video path")
    parser.add_argument('-o', '--output_video', help="output video path", default="output.mp4")
    parser.add_argument('--fps', help="fps", default=5)
    parser.add_argument('--configuration_file_path', help="path to parameter config file")
    args = parser.parse_args()
    main(args)
