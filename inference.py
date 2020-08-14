from tool.torch_utils import *
from tool.torch_utils import Yolov4Classifier
from classifier.model import Yolov4
from argparse import ArgumentParser

from tool.utils import load_class_names, plot_boxes_cv2
from config.dot_style_configuration import DotStyleConfiguration
from tool.util.nms import PostProcessing
from tool.util.generate_image_batch import get_image_batches
from tool.util.batch_transform import Transform
import cv2
from moviepy.editor import VideoFileClip

logging.basicConfig(level=os.getenv("LOG_LEVEL", logging.DEBUG))
LOGGER = logging.getLogger(__name__)


def create_directory(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        return True
    else:
        return False


def main(args):
    parameter_config = DotStyleConfiguration(args.configuration_file_path)
    weightfile = args.weights_file_path
    input_directory = args.input_dir

    n_classes = parameter_config.model_config.num_classes
    device = parameter_config.model_config.device

    model = Yolov4(yolov4conv137weight=None, n_classes=n_classes, inference=True)

    pretrained_dict = torch.load(weightfile, map_location=torch.device('cpu'))
    model.load_state_dict(pretrained_dict)

    if torch.cuda.is_available():
        model.cuda()
        device = torch.device(device)

    create_directory(args.output_dir)

    LOGGER.debug("device: ", device)
    confidence_threshold = parameter_config.model_config.confidence_threshold
    nms_threshold = parameter_config.model_config.nms_threshold
    height, width = (parameter_config.model_config.image_height, parameter_config.model_config.image_width)
    batch_size = parameter_config.batch_size

    class_map = parameter_config.class_map
    original_class_map = {int(key): value for key, value in class_map.items()}
    class_map = {
        key: value for key, value in original_class_map.items()
    }
    print(class_map)

    post_processing = PostProcessing(confidence_threshold, nms_threshold)
    transform = Transform(height, width)
    yolov4 = Yolov4Classifier(post_processing, transform, confidence_threshold, nms_threshold, device, height, width)

    height, width = (parameter_config.model_config.image_height, parameter_config.model_config.image_width)

    classification_worker = ObjectClassificationWorker(batch_size=batch_size,
                                                       classifier=yolov4,
                                                       model=model,
                                                       input_directory=input_directory,
                                                       width=width,
                                                       height=height,
                                                       class_map=class_map)
    classification_worker.worker()


class ObjectClassificationWorker:
    def __init__(self, batch_size, classifier, model, input_directory, width, height, class_map):
        self.batch_size = batch_size
        self.model = model
        self.input_directory = input_directory
        self.width = width
        self.height = height
        self.classifier = classifier
        self.class_map = class_map

    def preprocess(self, original_image_path_list):
        processed_image_batch = list()
        original_sizes = list()
        original_image_batch = list()
        for original_image_path in original_image_path_list:
            original_image = cv2.imread(original_image_path)
            original_image_batch.append(original_image)
            resized_image = cv2.resize(original_image, (self.width, self.height))
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            processed_image_batch.append(resized_image)
            original_sizes.append([original_image.shape[0], original_image.shape[1]])
        return original_sizes, original_image_batch, processed_image_batch

    def worker(self):
        original_image_batch_list = get_image_batches(self.input_directory, self.batch_size)
        for original_image_batch in original_image_batch_list:
            preprocessed_output = self.preprocess(original_image_batch)
            original_sizes, original_image_batch, processed_image_batch = zip(*preprocessed_output)
            boxes = self.classifier.predict_batch(self.model, processed_image_batch)
            for idx, box in enumerate(boxes):
                output_filename = processed_image_batch[idx].split('/')[-1]
                savename = args.output_dir + '/' + output_filename
                img = plot_boxes_cv2(original_image_batch[idx], box, savename, self.class_map)

        return


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_dir', help="input directory path containing .jpg images")
    parser.add_argument('--output_dir', help="output dir to save images", required=True)
    parser.add_argument('--configuration_file_path', help="path to parameter config file")
    parser.add_argument('-w', '--weights_file_path', help="path to model weights file, yolov4.pth")
    args = parser.parse_args()
    main(args)
