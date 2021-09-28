import os
import pandas as pd

from pipeline.steps import PipelineStep, SampleFilter


class HaririData(PipelineStep):

    def _execute(self, pheno_file: str, data_path: str, *args, **kwargs):
        df = pd.read_csv(os.path.join(data_path, 'hariri_complete.csv'), index_col='Proband')

        # ToDo: check delimiter
        df_clinical = pd.read_csv(pheno_file, index_col='Proband', na_values=[-99, 'nan'], sep=';', decimal=',')
        
        df = df.join(df_clinical, how='inner')

        df, _ = SampleFilter().apply_filter(self.pipeline.filter_name, df)

        self.pipeline.X_names = ['path_hariri_con_0001.img']
        self.pipeline.df = df

        text = """
        ### Hariri fMRI Data
        Loading data from 
        {}

                """.format(os.path.join(data_path, 'hariri_complete.csv'))
        self.fid.write(text)
