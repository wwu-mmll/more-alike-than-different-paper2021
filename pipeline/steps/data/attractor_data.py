import pandas as pd
import numpy as np

from pipeline.steps import PipelineStep
from pipeline.steps import SampleFilter


class AttractorData(PipelineStep):
    def _execute(self, attractor_filename: str,
                 attractors: list,
                 include_hc: bool = False,
                 *args, **kwargs):
        # attractor preprocessing
        df = pd.read_csv(attractor_filename, index_col='Subject_ID')

        # find top 5 attractors
        u, n = np.unique(df['attractor_number'], return_counts=True)
        sorted_att = np.argsort(n)[::-1]

        # remove rest
        attr_to_keep = set(sorted_att[attractors])
        attr_to_remove = set(sorted_att) - attr_to_keep
        for a in attr_to_remove:
            df['attractor_number'][df['attractor_number'] == u[a]] = np.nan

        attractor = df[['attractor_number']]

        # get data
        for_df = self.pipeline.df.copy()

        # split groups
        mdd = for_df[for_df['Group'] == 'MDD']
        hc = for_df[for_df['Group'] == 'HC']

        # join frames
        mdd = mdd.join(attractor)
        mdd.dropna(subset=['attractor_number'], inplace=True)

        # prepare hc frame
        hc['attractor_number'] = np.zeros(hc.shape[0])

        # concatenate hc and mdd
        if include_hc:
            final_df = pd.concat([hc, mdd], axis=0)
        else:
            final_df = mdd

        final_df['attractor_number'] = final_df['attractor_number'].astype("category")
        df, filter_variable = SampleFilter().apply_filter(self.pipeline.filter, final_df)
        self.pipeline.df = df
