import os
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_context("notebook", font_scale=1.4, rc={"lines.linewidth": 2.5})

from pipeline.steps import PipelineStep


class RocCurve(PipelineStep):

    def _execute(self, group_variable: str, n_best: int = 1, *args, **kwargs):
        df = self.pipeline.df
        best = self.pipeline.X_best
        best_names = self.pipeline.X_names[best][:n_best]
        self.plot_roc(df.loc[:, best_names].to_numpy(), df[group_variable].astype('category').cat.codes.to_numpy(),
                      self.pipeline.result_dir, fid=self.pipeline.fid)

    @staticmethod
    def plot_roc(X: np.ndarray, y: np.ndarray, results_folder: str = None, fid=None):
        # parameters
        lw = 2
        alpha = 1
        alpha_reduced = 0.5
        plt.figure(dpi=300)

        # check dimensionality
        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)

        y_dummy = pd.get_dummies(y).to_numpy()

        # loop through all variables and plot individual ROC curves within one plot
        for variable in range(X.shape[1]):
            current_X = X[:, variable].reshape(-1, 1)

            # drop NaNs
            filter = np.squeeze(~np.isnan(current_X))
            y_dummy_current = y_dummy[filter, :]
            current_X = current_X[filter, :]

            lr = LogisticRegression(penalty='none', class_weight='balanced')
            lr.fit(current_X, y[filter])
            y_score = lr.predict_proba(current_X)

            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(2):
                fpr[i], tpr[i], _ = roc_curve(y_dummy_current[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_dummy_current.ravel(), y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            if variable == 0:
                plt.plot(fpr[0], tpr[0], color=(210 / 256, 105 / 256, 30 / 256, alpha),
                         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
            else:
                plt.plot(fpr[0], tpr[0], color=(210 / 256, 105 / 256, 30 / 256, alpha_reduced), lw=lw / 2)

        # add reference line
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

        # so some styling
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")

        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        if results_folder:
            plt.savefig(os.path.join(results_folder, 'roc.png'))
        if fid:
            fid.write("\n## ROC\n")
            fid.write("![roc](roc.png)\n")
        plt.show()
