from pipeline.steps import PipelineStep
from multi_modality_datacreator.data_creator import run_graph_metrics_dti, run_graph_metrics_rs


class GraphMetricsRestingState(PipelineStep):

    def _execute(self, subjects: str = None, resting_state_path: str = None,
                 output_path: str = None, atlases: str = None, metrics: list = None,
                 n_processes: int = 1,
                 *args, **kwargs):
        run_graph_metrics_rs(sample_file=subjects,
                             resting_state_path=resting_state_path,
                             output_path=output_path,
                             atlases=atlases,
                             metrics=metrics,
                             n_processes=n_processes,
                             )


class GraphMetricsDTI(PipelineStep):

    def _execute(self, subjects: str = None, dti_parameter_file: str = None,
                 dti_nos_file: str = None, output_path: str = None, metrics: list = None,
                 n_processes: int = 1, *args, **kwargs):
        run_graph_metrics_dti(sample_file=subjects,
                              dti_parameter_file=dti_parameter_file,
                              dti_nos_file=dti_nos_file,
                              nos_threshold=3,
                              output_path=output_path,
                              metrics=metrics,
                              n_processes=n_processes)
