import os
from multi_modality_datacreator.data_creator import run_cat12

from pipeline.steps import PipelineStep


class Cat12Preprocessing(PipelineStep):

    def _execute(self, subjects: str, output_path: str,
                 cat12_path_batch1: str = None, cat12_path_batch2: str = None,
                 cat12_files_contain_sub_id: bool = False,
                 *args, **kwargs):
        os.makedirs(output_path, exist_ok=True)
        run_cat12(sample_file=subjects,
                  output_path=output_path,
                  cat12_path_batch1=cat12_path_batch1,
                  cat12_path_batch2=cat12_path_batch2,
                  cat12_files_contain_sub_id=cat12_files_contain_sub_id
                  )

        self.fid.write("Cat12 preprocessing done\n")
