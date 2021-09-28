import os
from datetime import datetime
from pipeline.steps import PipelineStep
from pipeline import constants


class Pipeline:
    """
    A pipeline itself is no part of a pipeline
    """
    def __init__(self, name: str = None, result_dir: str = None, filter_name: str = None):
        self.name = name
        self.result_dir = result_dir
        self.filter_name = filter_name
        self.constants = constants
        self.df = None
        self.X_names = None
        self.X_best = None
        self.fid = self._initialize_results()
        self.__pipeline_steps = []
        self.__post_hooks = []

    def execute(self):
        """
        Executes the pipeline
        If any exception is raised, the post hooks are called with the error message.
        If no exception is raised, the post hooks are called with a success message.
        :return:
        """
        self.summary()
        for (step, params) in self.__pipeline_steps:
            try:
                step.execute(**params)
            except Exception as e:
                print("pipeline failed")
                msg = "Pipeline \"{}\" failed in step {} with message {}".format(self.name, step.name, e)
                self.__send_hooks(msg)
                raise e
        print(75 * '*')
        print("Pipeline successfully completed.\n")
        message = "Pipeline \"{}\" succeeded".format(self.name)
        self.__send_hooks(message)
        self._write_results()
        self.fid.close()

    def _write_results(self):
        for step, _ in self.__pipeline_steps:
            log_file = os.path.join(self.result_dir, 'logs', step.name + '.log')
            if os.path.exists(log_file):
                fid = open(os.path.join(self.result_dir, 'logs', step.name + '.log'))
                txt = fid.read()
                self.fid.write(txt)
                fid.close()

    def send_warning(self, warning: str = None, step: PipelineStep = None):
        """
        Function to send a warning to the user

        :param warning: Warning string
        :param step:
        :return:
        """
        if warning is None:
            return
        warning = "Step {} raised warning {}".format(step.name, warning)
        print(warning)
        self.__send_hooks(warning)

    def __send_hooks(self, message):
        """
        Execute the post hooks

        :param message: Message to send
        """
        for hook in self.__post_hooks:
            try:
                hook(message)
            except Exception as e:
                print("Could not execute hook {}\nError: {}".format(hook, e))

    def add_post_hook(self, function):
        """
        A post hook is a python function intended to notify the user.

        Example: The user can get informed by a slack message when the pipeline succeeds.
        The post hook has to be callable or it will fail.


        :param function:
        :return:
        """
        self.__post_hooks.append(function)

    def __iadd__(self, other: tuple = None):
        """
        Add finder_pipeline elements with the format (PipelineElement, params_dict).

        :param other: Tuple of elements to add.
        :return:
        """
        if not isinstance(other, tuple) or other is None:
            raise ValueError("The added element has to be a tuple")
        if not isinstance(other[0], PipelineStep):
            raise ValueError("The added element has to contain a finder_pipeline step")
        if isinstance(other[0], Pipeline):
            raise ValueError("Pipelines can not contain pipelines")
        self.__pipeline_steps.append(other)
        return self

    def summary(self):
        """
        Print a simple summary of the current pipeline

        :return:
        """
        print(75*'*')
        print("{}".format(self.name))
        print(75 * '*')
        print("Pipeline setup:")
        print('  - sample filter: {}'.format(self.filter_name))
        for item in self.__pipeline_steps:
            print("  - {}".format(item[0].name))
        print(75 * '*')
        print("Starting execution")

    def _initialize_results(self):
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(os.path.join(self.result_dir, 'logs'), exist_ok=True)
        file = open(os.path.join(self.result_dir, self.constants['results_file']), "w")
        file.write("# Results: {}\n".format(self.name))
        now = datetime.now()
        file.write("Time: {}\n".format(now.strftime("%d/%m/%Y, %H:%M:%S")))
        return file
