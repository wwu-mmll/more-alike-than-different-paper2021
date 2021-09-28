import os

from multi_modality_datacreator.data_creator import run_freesurfer

from pipeline.steps import PipelineStep


class FreesurferPreprocessing(PipelineStep):

    def _execute(self, subjects: str, freesurfer_path: str = None, output_path: str = None,
                  *args, **kwargs):
        run_freesurfer(reference_path=freesurfer_path,
                       sample_path=os.path.dirname(subjects),
                       sample_filename=os.path.basename(subjects),
                       output_path=output_path)

        self.fid.write("Freesurfer preprocessing done\n")
