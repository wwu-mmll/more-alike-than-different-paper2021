import os
from multi_modality_datacreator.data_creator import run_hariri

from pipeline.steps import PipelineStep


class HaririPreprocessing(PipelineStep):

    def _execute(self, subjects: str, output_path: str, voxel_size: int = 3, create_nifti_pickles: bool = False, *args, **kwargs):
        os.makedirs(output_path, exist_ok=True)
        run_hariri(sample_file=subjects, output_path=output_path,
                   voxel_size=voxel_size,
                   create_nifti_pickles=create_nifti_pickles)

        self.fid.write("Hariri preprocessing done\n")
