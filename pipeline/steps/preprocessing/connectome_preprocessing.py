from pipeline.steps import PipelineStep
from multi_modality_datacreator.data_creator import run_resting_state, run_dti


class RestingStatePreprocessing(PipelineStep):

    def _execute(self, sample_file: str = None,
                 resting_state_path: str = None, output_path: str = None,
                 atlases: list = ['Schaefer100-17'], *args, **kwargs):
        run_resting_state(sample_file=sample_file,
                          resting_state_path=resting_state_path,
                          output_path=output_path,
                          atlases=atlases)


class DTIPreprocessing(PipelineStep):

    def _execute(self, sample_file: str = None, dti_file: str = None,
                 output_path: str = None, parameters: str = None,
                 *args, **kwargs):
        run_dti(sample_file=sample_file,
                dti_file=dti_file,
                output_path=output_path,
                parameters=parameters)
