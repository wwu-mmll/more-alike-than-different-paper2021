import os
import pandas as pd

from pipeline.steps import PipelineStep


class PRSData(PipelineStep):

    def _execute(self, data_path: str, prs_variants: list = None, *args, **kwargs):
        df = pd.read_csv(os.path.join(data_path, 'prs_complete_norel.csv'), index_col='Proband')

        if prs_variants[0].lower() == 'all':
            self.pipeline.X_names = [prs for prs in df.columns if 'PRS' in prs]
        else:
            self.pipeline.X_names = list()
            for prs in prs_variants:
                self.pipeline.X_names.extend([var for var in df.columns if prs in var])

        self.pipeline.df = df
