import os
import pandas as pd

from pipeline.steps import PipelineStep, SampleFilter


class ClinicalData(PipelineStep):

    def _execute(self, sample_file: str, variables: list = None, *args, **kwargs):
        df_clinical = pd.read_csv(sample_file, index_col='Proband', na_values=[-99, 'nan', -2], sep=';', decimal=',')

        df, _ = SampleFilter().apply_filter(self.pipeline.filter_name, df_clinical)

        # get X names
        var_list = list()
        if variables:
            for var in variables:
                if "*" in var:
                    existing_vars = [s for s in df.columns if var.replace("*", "") in s]
                    var_list.extend(existing_vars)
                else:
                    var_list.append(var)
            self.pipeline.X_names = var_list

        self.pipeline.df = df

