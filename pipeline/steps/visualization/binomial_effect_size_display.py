import os
import pandas as pd
from scipy.stats import pointbiserialr

from pipeline.steps import PipelineStep


class BinomialEffectSizeDisplay(PipelineStep):

    def _execute(self, group_variable: str = 'Group', *args, **kwargs):
        X_names = self.pipeline.X_names
        X_best = self.pipeline.X_best
        self.binomial_effect_size_display(self.pipeline.df, X_names[X_best[0]], group_variable,
                                          os.path.join(self.pipeline.result_dir, 'besd.csv'),
                                          fid=self.pipeline.fid)

    @staticmethod
    def binomial_effect_size_display(df: pd.DataFrame, x: str, group: str, filename: str = None, fid=None):
        if fid:
            fid.write("\n## Binomial Effect Size Display\n")
        group_names = df[group].unique()

        # calculate point biserial correlation
        if df[group].dtype == 'object':
            df[group] = df[group].astype('category')
            df[group] = df[group].cat.codes
        # drop nans
        if df[group].hasnans or df[x].hasnans:
            n_subjects = df.shape[0]
            df.dropna(subset=[group, x], inplace=True)
            fid.write("Dropping {} subjects because of NaNs\n\n".format(n_subjects - df.shape[0]))
        fid.write("{} subjects used for BESD\n\n".format(df.shape[0]))

        r, _ = pointbiserialr(df[group], df[x])

        # calculate success rates
        control = 0.5 - r / 2
        treatment = 0.5 + r / 2

        # calculate remaining cells
        results = pd.DataFrame({'Groups': [group_names[0], group_names[1], 'Total'],
                                'Pred {}'.format(group_names[0]): [control, treatment, 100],
                                'Pred {}'.format(group_names[1]): [treatment, control, 100],
                                'Total': [100, 100, 200]})
        print(results)
        if filename:
            results.to_csv(filename)
        if fid:
            fid.write(results.to_markdown())
