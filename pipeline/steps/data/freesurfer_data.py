import os
import pandas as pd
import numpy as np
from scipy import stats

from pipeline.steps import PipelineStep, SampleFilter


class FreesurferData(PipelineStep):

    def _execute(self, data_path: str, pheno_file: str, remove_outlier: bool = True, remove_nan: str = "impute",
                 use_only_acute: bool = False, *args, **kwargs):
        self.remove_outlier = remove_outlier
        self.remove_nan = remove_nan
        self.pheno_file = pheno_file
        self.freesurfer_file = os.path.join(data_path, 'freesurfer_complete.csv')

        # load phenotype data (questionnaires, demographics and so on)
        pheno = pd.read_csv(self.pheno_file, sep=';', decimal=',', index_col='Proband', na_values=-99)

        # load freesurfer data
        fs_data = pd.read_csv(self.freesurfer_file, index_col='Proband', na_values=-99)
        fs_data.columns = [string.replace('.', '_') for string in fs_data.columns]
        measure_names = fs_data.columns[1:]
        df = pheno.join(fs_data, rsuffix='_r', how='inner')
        df.dropna(subset=['Group'], inplace=True)

        # filter data
        df, _ = SampleFilter().apply_filter(self.pipeline.filter_name, df)

        # handle outliers
        df = self._find_outlier(df, columns=measure_names, threshold=3, strategy='impute')
        self.pipeline.df = df
        measures_df = pd.DataFrame({'X_names': measure_names})
        measures_df.to_csv(os.path.join(data_path, 'X_names.csv'))
        self.pipeline.X_names = measure_names
        self._write()

    @staticmethod
    def _find_outlier(df: pd.DataFrame, columns: list, threshold: float, strategy: str = 'impute'):
        clean_df = df.copy()
        data = clean_df.loc[:, columns].values
        z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
        z_filter = z_scores > threshold
        print("Found {} outlier values (on average {} per feature).".format(np.sum(z_filter),
                                                                            np.mean(np.sum(z_filter, axis=0))))

        for i, col in enumerate(columns):
            clean_df.loc[z_filter[:, i], col] = np.nan

        if strategy.lower() == 'drop':
            print("Dropping outliers")
            clean_df = clean_df.dropna()
        elif strategy.lower() == 'impute':
            data = clean_df.loc[:, columns].values
            means = np.nanmean(data, axis=0)
            isnan = np.isnan(data)
            for i, col in enumerate(columns):
                clean_df.loc[isnan[:, i], col] = means[i]
        return clean_df

    def _write(self):
        text = """
### Freesurfer Data
Loading data from 
{}

and

{}

### Outlier and Imputation
Removing outlier: {}

Imputation strategy: {}

        """.format(self.pheno_file, self.freesurfer_file, self.remove_outlier, self.remove_nan)
        self.fid.write(text)
