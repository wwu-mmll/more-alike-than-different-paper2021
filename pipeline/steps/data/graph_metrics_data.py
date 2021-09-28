import os
import pandas as pd

from pipeline.steps import PipelineStep, SampleFilter


class GraphMetricsRSData(PipelineStep):

    def _execute(self, pheno_file: str, data_path: str, atlas: list = None, *args, **kwargs):
        df = None
        if atlas is None:
            df = pd.read_csv(os.path.join(data_path, 'gm_rs_complete.csv'), index_col='Subject_ID')
        else:
            for i, atlas_name in enumerate(atlas):
                atlas_df = pd.read_csv(os.path.join(data_path, '{}.csv'.format(atlas_name)),
                                       index_col='Proband')
                if i == 0:
                    df = atlas_df
                else:
                    df = pd.merge(atlas_df, df, how='outer')

        df.columns = df.columns.str.replace('-', '_')
        self.pipeline.X_names = df.columns.tolist()

        # get RS SPSS file
        # ToDo: Check delimiter
        df_pheno = pd.read_csv(pheno_file, index_col='Proband', na_values=[-99, 'nan', -99.0], sep=';', decimal=',')

        # merge dataframes
        df = df_pheno.join(df, how='inner')
        df.dropna(subset=[self.pipeline.X_names[0]], inplace=True)

        df, _ = SampleFilter().apply_filter(self.pipeline.filter_name, df)

        self.pipeline.df = df


class GraphMetricsDTIData(PipelineStep):

    def _execute(self, pheno_file: str, data_path: str, *args, **kwargs):
        pheno_df = pd.read_csv(pheno_file, index_col='Proband', decimal=',', sep=';', na_values=['nan', -99])

        gm = pd.read_csv(os.path.join(data_path, 'graph_metrics_dti_complete.csv'), index_col='Proband')

        df = pheno_df.merge(gm, how='inner', on='Proband')

        self.pipeline.X_names = gm.columns.tolist()

        df, _ = SampleFilter().apply_filter(self.pipeline.filter_name, df)
        self.pipeline.df = df
