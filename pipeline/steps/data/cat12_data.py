import os
import pandas as pd

from pipeline.steps import PipelineStep
from pipeline.steps import SampleFilter


class Cat12Data(PipelineStep):

    def _execute(self, data_path: str, pheno_file: str, smoothed: bool = True, *args, **kwargs):
        pheno = pd.read_csv(pheno_file, sep=';', decimal=',', index_col='Proband', na_values=-99)

        cat12_file = os.path.join(data_path, 'cat12_complete.csv')
        cat12_data = pd.read_csv(cat12_file, index_col='Proband',
                                 na_values=[-99, 'nan'])

        # join both dataframes
        cat12_data = cat12_data.join(pheno, rsuffix='_r')

        # create actual absolute filename
        cat12_data['filesep'] = ['/'] * cat12_data.shape[0]
        X_names = ['gray_matter_absolute_path']
        if smoothed:
            cat12_data[X_names[0]] = cat12_data['path_mwp1'] + cat12_data['filesep'] + 's8' + cat12_data['filename_mwp1']
        else:
            cat12_data[X_names[0]] = cat12_data['path_mwp1'] + cat12_data['filesep'] + cat12_data['filename_mwp1']

        # filter sample
        cat12_data, _ = SampleFilter().apply_filter(self.pipeline.filter_name, cat12_data)
        self.pipeline.df = cat12_data

        # fix TIV
        tiv = [float(str(tiv).replace(',', '.')) for tiv in self.pipeline.df['TIV']]
        self.pipeline.df['TIV'] = tiv
        self.pipeline.X_names = X_names

        # check for NaNs
        if self.pipeline.df[X_names[0]].hasnans:
            print("Found NaNs in Cat12 data. Dropping NaNs.")
            self.pipeline.df.dropna(subset=[X_names[0]], inplace=True)

        text = """
        ### Cat12 Data
        Loading data from 
        {}

                """.format(cat12_file)
        self.fid.write(text)
