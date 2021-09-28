import os
import pandas as pd
from abc import ABC

from pipeline import constants


class PipelineStep(ABC):
    """
    Abstract wrapper for finder_pipeline steps
    """

    def __init__(self,
                 name: str = None,
                 pipeline_name: str = None,
                 result_dir: str = None,
                 pipeline=None):
        """
        Initialize a pipeline step

        :param name: name of the current finder_pipeline step
        :param pipeline_name: name of the current finder_pipeline
        :param result_dir: directory for results of the current finder_pipeline
        :param metadata: csv file containing the metadata
        :param pipeline: The finder_pipeline containing this element
        """
        if not os.path.isdir(result_dir):
            raise ValueError("ResultsDir is not existing")

        self.result_dir = result_dir
        self.name = name
        self.pipeline_name = pipeline_name
        self.constants = constants
        self.pipeline = pipeline
        self.fid = None

    def execute(self, skip: bool = False, *args, **kwargs):
        """
        Execute the current finder_pipeline step
        :return: Result of the current step
        """
        if skip:
            print("skipping step: {}".format(self.name))
            return
        print("running step: {}".format(self.name))
        fid_name = os.path.join(self.result_dir, 'logs', self.name + '.log')
        self.fid = open(fid_name, "w")
        self.fid.write("\n## {}\n".format(self.name))
        print("Writing logs to: {}".format(fid_name))
        res = self._execute(*args, **kwargs)
        self.fid.close()
        return res

    def _execute(self, *args, **kwargs):
        """
        This function has to be overwritten by finder_pipeline steps
        :return: Result of the calculation
        """
        raise NotImplementedError("Subclasses must provide this function")

    def _warn(self, message):
        """
        Function to send a warning to the user


        :param message: Warning message
        """
        self.pipeline.send_warning(message, self)
