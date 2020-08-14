from tool.torch_utils import *
from tool.torch_utils import Yolov4Classifier
from classifier.model import Yolov4
from argparse import ArgumentParser

from tool.utils import load_class_names, plot_boxes_cv2
from config.dot_style_configuration import DotStyleConfiguration
from tool.util.nms import PostProcessing
from tool.util.batch_transform import Transform
import cv2
from moviepy.editor import VideoFileClip

logging.basicConfig(level=os.getenv("LOG_LEVEL", logging.DEBUG))
LOGGER = logging.getLogger(__name__)

processed_images = list()
original_sizes = list()
original_images = list()


def create_directory(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        return True
    else:
        return False


def main(args):
    global frame_count
    global model
    global use_cuda
    global parameter_config
    global yolov4
    global class_map

    parameter_config = DotStyleConfiguration(args.configuration_file_path)

    weightfile = args.weights_file_path

    n_classes = parameter_config.model_config.num_classes
    input_video = args.input_video
    output_video = args.output_video

    model = Yolov4(yolov4conv137weight=None, n_classes=n_classes, inference=True)

    pretrained_dict = torch.load(weightfile, map_location=torch.device('cpu'))
    model.load_state_dict(pretrained_dict)

    if torch.cuda.is_available():
        use_cuda = True
        model.cuda()
        device = torch.device(parameter_config.model_config.device)
    else:
        use_cuda = False
        device = "cpu"

    create_directory(args.output_dir)

    LOGGER.debug("device: ", device)
    frame_count = 0
    confidence_threshold = parameter_config.model_config.confidence_threshold
    nms_threshold = parameter_config.model_config.nms_threshold
    height, width = (parameter_config.model_config.image_height, parameter_config.model_config.image_width)

    class_map = parameter_config.class_map
    original_class_map = {int(key): value for key, value in class_map.items()}
    class_map = {
        key: value for key, value in original_class_map.items()
    }
    print(class_map)

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
    global class_map
    height, width = (parameter_config.model_config.image_height, parameter_config.model_config.image_width)
    resized_image = cv2.resize(original_image, (width, height))
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    batch_size = parameter_config.batch_size

    processed_images.append(resized_image)
    original_sizes.append([original_image.shape[0], original_image.shape[1]])
    original_images.append(original_image)
    img = original_image

    if batch_size == 1 or (frame_count > 0 and frame_count % batch_size == 0):
        boxes = yolov4.predict_batch(model, processed_images)
        for idx, box in enumerate(boxes):
            savename = args.output_dir + '/' + str(frame_count - batch_size + idx) + '.jpg'
            img = plot_boxes_cv2(original_images[idx], box, savename, class_map)
        processed_images.clear()
        original_sizes.clear()
        original_images.clear()

    frame_count += 1

    return img


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_video', help="input video path")
    parser.add_argument('-o', '--output_video', help="output video path", default="output.mp4")
    parser.add_argument('--output_dir', help="output dir to save images", required=True)
    parser.add_argument('--fps', help="fps", default=5)
    parser.add_argument('--configuration_file_path', help="path to parameter config file")
    parser.add_argument('-w', '--weights_file_path', help="path to model weights file, yolov4.pth")
    args = parser.parse_args()
    main(args)
