import os
import pandas as pd

from pipeline.steps.visualization.classification_effect_size_plot_matplotlib import ClassificationEffectSizePlotMatplotlib

from pipeline.steps import PipelineStep


class PipelinePlot(PipelineStep):

    def _execute(self, group_variable: str, *args, **kwargs):
        df = self.pipeline.df
        if self.pipeline.X_best is None:
            best = pd.read_csv(os.path.join(self.result_dir, 'X_best.csv'))['X_best'].tolist()
        else:
            best = self.pipeline.X_best
        best_names = self.pipeline.X_names[best[0]]
        aov = pd.read_csv(os.path.join(self.pipeline.result_dir, 'anova_results.csv'))
        residuals = pd.read_csv(os.path.join(self.pipeline.result_dir, 'residuals.csv'), index_col='Proband')

        # drop NaNs
        df = df.dropna(subset=[best_names])

        plotter = ClassificationEffectSizePlotMatplotlib(df=df,
                                                         residuals=residuals,
                                                         group_name=group_variable,
                                                         variable_name=best_names,
                                                         partial_eta_squared=aov.loc[1, 'np2'],
                                                         partial_eta_squared_upper=aov.loc[1, 'np2_BCI_high'],
                                                         partial_eta_squared_lower=aov.loc[1, 'np2_BCI_low'])
        plotter.plot(filename=os.path.join(self.pipeline.result_dir, 'effect_size_plot.png'),
                     result_dir=self.pipeline.result_dir)

        self.fid.write("\n![plot](effect_size_plot.png)\n")
