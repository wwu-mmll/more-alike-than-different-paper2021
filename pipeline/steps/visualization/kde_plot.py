import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
sns.set_context("notebook", font_scale=1.4, rc={"lines.linewidth": 2.5})

from pipeline.steps import PipelineStep


class KdePlot(PipelineStep):

    def _execute(self, group_variable: str, *args, **kwargs):
        best_measure = self.pipeline.X_names[self.pipeline.X_best[0]]
        self.plot(self.pipeline.df,
                  best_measure,
                  group_variable,
                  os.path.join(self.pipeline.result_dir, 'kde.png'),
                  self.pipeline.fid)

    def plot(self, df: pd.DataFrame, x: str, group: str, filename: str = None,
             fid=None):
        plt.figure(dpi=300)
        ax = sns.kdeplot(data=df, x=x, hue=group,
                         fill=True, common_norm=False, palette="muted",
                         alpha=.5, linewidth=1.5)

        # display CLEST
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        ymax = ylim[1]
        xmax = np.mean(xlim)
        cless = round(self.common_language_effect_size_statistic(df, x, group) * 100)
        ax.text(xmax, ymax, 'CLESS = {}%'.format(cless), ha='center', fontsize=12)

        # some styling
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        if filename:
            plt.savefig(filename)

        if fid:
            fid.write("\n## KDE Plot\n")
            fid.write("![kde](kde.png)\n")
        plt.show()

    @staticmethod
    def common_language_effect_size_statistic(df: pd.DataFrame, x: str, group: str):
        group_names = df[group].unique()
        control = df[df[group] == group_names[0]]
        treatment = df[df[group] == group_names[1]]
        var_pooled = np.sqrt(np.var(control[x]) + np.var(treatment[x]))
        diff = np.mean(control[x]) - np.mean(treatment[x])
        z = (0 - np.abs(diff)) / var_pooled
        p = 1 - norm.cdf(z)
        return p