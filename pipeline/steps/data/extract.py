import os
import pandas as pd
from pipeline.steps import PipelineStep


class Extractor(PipelineStep):

    def _execute(self, *args, **kwargs):
        if self.pipeline.X_best is None:
            best = pd.read_csv(os.path.join(self.result_dir, 'X_best.csv'))['X_best'].tolist()
        else:
            best = self.pipeline.X_best
        best_name = self.pipeline.X_names[best[0]]

        peak_variable = self.pipeline.df[best_name]

        peak_variable.to_csv(os.path.join(self.result_dir, 'peak_variable.csv'))
