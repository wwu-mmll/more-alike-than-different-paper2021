#!/usr/bin/env python
import subprocess
import argparse
import os
from shutil import copyfile

import pipeline.steps as steps
from pipeline import ConfigReader, Pipeline


def main_pipeline(config_file: str = None) -> None:
    """
    Runs the main finder_pipeline with all defined steps
    :param config_file: path to the config file
    :return:
    """

    # First step: try to read the current config
    # This is the only step that can not be changed
    config = ConfigReader.execute(config_file)

    # add partials to config
    for step_name, step_config in config.items():
        if 'partial' in step_config.keys():
            config[step_name] = ConfigReader.execute(step_config['partial'])

    # Second step: Build the pipeline using the provided steps
    pipeline_name = config['setup']['analysis_name']
    filter_names = config['setup']['filter_name']
    if not isinstance(filter_names, list):
        if isinstance(filter_names, dict) and 'partial' in filter_names.keys():
            filter_names = ConfigReader.execute(filter_names['partial'])['filter_name']
        else:
            filter_names = [filter_names]

    for filter_name in filter_names:
        result_dir = os.path.join(config['setup']['result_dir'], pipeline_name, filter_name.replace('filter_', ''))
        os.makedirs(result_dir, exist_ok=True)
        # now copy config file to result dir (for documentation purposes)
        copyfile(config_file, os.path.join(result_dir, os.path.basename(config_file)))

        pipe = Pipeline(pipeline_name, result_dir, filter_name)
        for key in config.keys():
            if key in ['setup']:
                # the setup key contains generic information
                continue
            details = config[key]
            current_obj = getattr(steps, details['class'])(key, pipeline_name, result_dir, pipe)

            if 'params' in details and details['params'] is not None:
                pipe += (current_obj, details['params'])
            else:
                pipe += (current_obj, {})

        # Third step: execute the finder_pipeline
        pipe.execute()


if __name__ == '__main__':
    def file_path(path):
        if os.path.isfile(path):
            return path
        else:
            raise argparse.ArgumentTypeError("conf_file:{} is not a valid file path".format(path))

    parser = argparse.ArgumentParser(description="MDD vs HC Analysis")
    parser.add_argument('conf_file', type=file_path, help="path for the config.yaml file")
    args = parser.parse_args()
    main_pipeline(args.conf_file)
