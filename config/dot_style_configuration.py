# TODO: delete after getting dependant packages work

import yaml
from num2words import num2words


class DotStyleConfiguration(dict):
    """
    Takes in a path to a yaml file and converts the structure to a dot-style configuration.

    You can also use this configuration as a dictionary of dictionaries if dot-style isn't your forte.
    If you'd like to mix and match then this is also fine.

    Example:
        model_config:
          num_classes: 81
          model_weights: ./weights/yolact_im700_54_800000.pth
          eval_mask_branch: yes
          max_size: 700
          score_threshold: .2
          top_k: 200
          transform_device: cpu
          feature_select: true
          nested_features:
            feature_1: []
            feature_2: 10


        config = DotStyleConfiguration("path/to/yaml/file")

        config.model_config.num_classes
        >> 81
        config.model_config.nested_features.feature_1
        >> []
        config.model_config.nested_features.feature_2
        >> 10
        config.model_config.nested_features
        {"feature_1": [], "feature_2": 10}

    Equal Example:
        model_config:
        num_classes: 81
        model_weights: ./weights/yolact_im700_54_800000.pth
        eval_mask_branch: yes
        max_size: 700
        score_threshold: .2
        top_k: 200
        transform_device: cpu
        feature_select: true
        nested_features:
            feature_1: []
            feature_2: 10


        config = DotStyleConfiguration("path/to/yaml/file")

        config["model_config"].num_classes
        >> 81
        config["model_config"]["nested_features"]["feature_1"]
        >> []
        config["model_config"]["nested_features"]["feature_2"]
        >> 10
        config["model_config"].nested_features
        >> {"feature_1": [], "feature_2": 10}

    Examples when this DotStyleConfiguration works but is modified:

        class_map: # dictionary of classes we only care about (CART classes)
          1: bicycle
          2: car
          3: motorcycle
          5: bus
          7: truck

        config = DotStyleConfiguration("path/to/yaml/file")

        config.class_map.one
        >> bicycle

    """

    def __init__(self, config_file: str = None, **kwargs):
        super(DotStyleConfiguration, self).__init__()

        if config_file is not None:
            with open(config_file) as f:
                config = yaml.safe_load(f)
        else:
            config = kwargs

        for (key, val) in config.items():
            if isinstance(val, dict):
                sanitized_dict = self._sanitize_dict(val)
                self[key] = DotStyleConfiguration(**sanitized_dict)
            else:
                self[key] = val

    def _sanitize_dict(self, dictionary: dict) -> dict:
        """
        Sanitizes the dictionary of any integer values and converts them to their literals:
            1 -> one
            42 -> forty_two
            1100 -> one_thousand_one_hundred

        If there is nothing to sanitize then a new dictionary with the same values will be returned.

        :param dictionary: dictionary to sanitize
        :return: sanitized dictionary
        """
        sanitized_dict = dict()

        for (k, v) in dictionary.items():
            if isinstance(k, int):
                k = self._number_to_literal(k)

            sanitized_dict.update({k: v})

        return sanitized_dict

    def _number_to_literal(self, number: int) -> str:
        """
        Weird replace chaining to remove commas and hyphens from a number to use it as a key in a dictionary.
        :param number: Any number
        :return: The number transformed into a string literal
        """
        literal_number = num2words(number)
        literal_number = (
            literal_number.replace(",", "").replace("-", "_").replace(" ", "_")
        )
        return literal_number

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(DotStyleConfiguration, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(DotStyleConfiguration, self).__delitem__(key)
        del self.__dict__[key]
