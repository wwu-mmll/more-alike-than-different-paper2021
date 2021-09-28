import os
import yaml


class ConfigReader:
    """
    Config reader is no pipeline step.
    This would cause bootstrapping problems
    """

    @staticmethod
    def execute(config_path: str = None):
        """
        This function tries to load the configuration from a yaml file
        :param config_path: path to the configuration file (yaml)
        :return: configuration for the pipeline
        """
        if config_path is None:
            raise ValueError("No config file provided")
        if not os.path.isfile(config_path):
            raise ValueError("Could not find config file in path provided")

        stream = open(config_path, 'r')
        return yaml.load(stream, Loader=yaml.SafeLoader)
