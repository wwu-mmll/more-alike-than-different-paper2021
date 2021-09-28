import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import font_manager
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, precision_score, balanced_accuracy_score, recall_score
from math import erf
from scipy.stats import norm
from externals.gists.mclust.gaussian_mixture import Mclust
from externals.gists.plotting.fancy_box_plot import plot_hist_and_box


# set font
font_dir = "../pipeline/steps/visualization/fonts/"
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)
plt.rcParams['font.family'] = 'Share Tech'

SMALL_SIZE = 11
MEDIUM_SIZE = 13
BIGGER_SIZE = 13

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)


class ClassificationEffectSizePlotMatplotlib:

    def __init__(self, df: pd.DataFrame, residuals: pd.DataFrame, group_name: str, variable_name: str,
                 partial_eta_squared: float, partial_eta_squared_upper: float,
                 partial_eta_squared_lower: float):
        self.fig = None
        self.ax = None
        self.gs = None

        self.df = df.sort_values(by=[group_name], ascending=False)
        self.residuals = residuals.sort_values(by=[group_name], ascending=False)
        self.group_name = group_name
        self.variable_name = variable_name
        self.CLESS = None
        self.sensitivity = None
        self.specificity = None
        self.balanced_accuracy = None
        self.partial_eta_squared = partial_eta_squared
        self.partial_eta_squared_upper = partial_eta_squared_upper
        self.partial_eta_squared_lower = partial_eta_squared_lower
        self.overlap = None

        self.font_color = '#3D4142'

    def plot(self, result_dir: str = None, filename: str = None):
        self.fig = plt.figure(constrained_layout=False, figsize=(10, 5), dpi=400)
        self.gs = gridspec.GridSpec(ncols=3, nrows=4, figure=self.fig, hspace=0, wspace=0.3,
                                    width_ratios=[0.6, 0.3, 0.1],
                                    left=0.1, bottom=0.1)

        # plot roc
        self._plot_roc_curve()

        # plot hist and box
        self._plot_hist_and_box()

        # common language effect size statistic
        #self.common_language_effect_size_statistic()

        # calculate overlap
        self._calculate_overlap_on_residuals()

        # plot donuts
        self._plot_percentage_donut()

        # plot effect size
        self._plot_effect_size()

        if result_dir:
            res = pd.DataFrame({'Sensitivity': self.sensitivity,
                                'Specificity': self.specificity,
                                'CLESS': self.CLESS,
                                'BalancedAccuracy': self.balanced_accuracy,
                                'Partial Eta2': self.partial_eta_squared,
                                'Partial Eta2 Upper': self.partial_eta_squared_upper,
                                'Partial Eta2 Lower': self.partial_eta_squared_lower,
                                'Overlap': self.overlap},
                               index=[1])
            res.to_csv(os.path.join(result_dir, 'effect_size_results.csv'))

        if filename:
            plt.savefig(filename)

        plt.show()

    def _plot_percentage_donut(self):
        colors = ['#E6E7E8', '#2E86C1']

        metrics = {'Overlap': self.overlap, 'BACC': self.balanced_accuracy, 'Sensitivity': self.sensitivity,
                   'Specificity': self.specificity}

        cnt = 0
        for metric, value in metrics.items():
            values = [1 - value, value]
            ax = self.fig.add_subplot(self.gs[cnt, -1])
            ax.pie(values, wedgeprops=dict(width=0.25),
                   startangle=90, colors=colors)
            ax.set_title(metric, pad=-3)
            ax.text(0.5, 0.5, "{:.1f}%".format(value*100), fontsize=8,
                    verticalalignment='center', horizontalalignment='center',
                    transform=ax.transAxes)
            cnt += 1
        ax.text(0, -2, "(d)", fontsize=10, horizontalalignment='center')

    def _plot_hist_and_box(self):
        ax = self.fig.add_subplot(self.gs[:, 0])
        self.fig, ax1, ax2 = plot_hist_and_box(fig=self.fig, ax=ax, df=self.residuals,
                                               x="residuals", hue=self.group_name,
                                               title=self.variable_name)
        ax.set_title("(a)", fontsize=10)

    def _calculate_overlap_on_residuals(self):
        df = self.df
        unique_values = df[self.group_name].unique()
        x1 = self.residuals.loc[df[self.group_name] == unique_values[0], 'residuals'].values
        x2 = self.residuals.loc[df[self.group_name] == unique_values[1], 'residuals'].values
        mclust1 = Mclust("V", n_gaussians=1)
        mclust1.fit(x1)
        mu1 = np.asarray(mclust1.model.rx2['parameters'].rx2['mean'])[0]
        sigma1 = np.sqrt(np.asarray(mclust1.model.rx2['parameters'].rx2['variance'].rx2['sigmasq']))
        mclust2 = Mclust("V", n_gaussians=1)
        mclust2.fit(x2)
        mu2 = np.asarray(mclust2.model.rx2['parameters'].rx2['mean'])[0]
        sigma2 = np.sqrt(np.asarray(mclust2.model.rx2['parameters'].rx2['variance'].rx2['sigmasq']))
        self.overlap = self._normdist_calculate_overlap(mu1, sigma1, mu2, sigma2)[0]

    @staticmethod
    def _normdist_calculate_overlap(mu1, sigma1, mu2, sigma2):
        """
        Re-implementation of Distnorm.overlap() in Python 3.9 statistics module
        :param mu1:
        :param sigma1:
        :param mu2:
        :param sigma2:
        :return: distributional overlap
        """

        if (sigma2, mu2) < (sigma1, mu1):  # sort to assure commutativity
            sigma1, sigma2 = sigma2, sigma1
            mu1, mu2 = mu2, mu1

        X_var, Y_var = np.square(sigma1), np.square(sigma2)
        if not X_var or not Y_var:
            raise RuntimeError('overlap() not defined when sigma is zero')
        dv = Y_var - X_var
        dm = np.fabs(mu2 - mu1)

        if not dv:
            return 1.0 - erf(dm / (2.0 * sigma1 * np.sqrt(2.0)))
        a = mu1 * Y_var - mu2 * X_var
        b = sigma1 * sigma2 * np.sqrt(dm**2.0 + dv * np.log(Y_var / X_var))
        x1 = (a + b) / dv
        x2 = (a - b) / dv
        return 1.0 - (np.fabs(norm.cdf(x1, mu2, sigma2) - norm.cdf(x1, mu1, sigma1)) +
                      np.fabs(norm.cdf(x2, mu2, sigma2) - norm.cdf(x2, mu1, sigma1)))

    def _plot_roc_curve(self):
        X = self.residuals['residuals'].values
        X = X.reshape(-1, 1)
        y = self.df[self.group_name].astype('category').cat.codes.to_numpy()
        y_dummy = pd.get_dummies(y).to_numpy()
        current_X = X

        # drop NaNs
        filter = np.squeeze(~np.isnan(current_X))
        y_dummy_current = y_dummy[filter, :]
        y_current = y[filter]
        current_X = current_X[filter, :]

        use_logistic_regression = True
        if use_logistic_regression:
            lr = LogisticRegression(penalty='none', class_weight='balanced', max_iter=1000000)
            lr.fit(current_X, y[filter])
            y_pred = lr.predict(current_X)
            y_score = lr.predict_proba(current_X)
        else:
            # use mclust (Gaussian Mixture model) to predict class 1 versus class 2
            from externals.gists.mclust.gaussian_mixture import Mclust
            mclust = Mclust("V", n_gaussians=2)
            mclust.fit(current_X, y_current)
            y_pred = mclust.predict(current_X)

            # create inverse of cluster labels
            y_pred_inverse = y_pred.copy()
            y_pred_inverse[y_pred == 2] = 0

            # substract 1 because mclust will give cluster labels as 1 and 2, but pandas categories are 0 and 1
            y_pred -= 1

            y_score = mclust.predict_proba(current_X)
            bacc_temp1 = balanced_accuracy_score(y_true=y_current, y_pred=y_pred)
            bacc_temp2 = balanced_accuracy_score(y_true=y_current, y_pred=y_pred_inverse)
            if bacc_temp1 < bacc_temp2:
                y_pred = y_pred_inverse
                y_score[:, [0, 1]] = y_score[:, [1, 0]]

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(2):
            fpr[i], tpr[i], _ = roc_curve(y_dummy_current[:, i], y_score[:, i], drop_intermediate=True)
            roc_auc[i] = auc(fpr[i], tpr[i])


        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_dummy_current.ravel(), y_score.ravel(), drop_intermediate=True)
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # compute sensitivity and specificity
        unique, counts = np.unique(y_current, return_counts=True)
        self.sensitivity = recall_score(y_current, y_pred, labels=unique, pos_label=unique[0])
        self.specificity = recall_score(y_current, y_pred, labels=unique, pos_label=unique[1])
        self.balanced_accuracy = balanced_accuracy_score(y_true=y_current, y_pred=y_pred)

        # plot ROC curve
        ax = self.fig.add_subplot(self.gs[2:, 1])
        ax.plot(fpr['micro'], tpr['micro'], color='#2E86C1')
        ax.fill_between(fpr['micro'], tpr['micro'], color='#2E86C1', alpha=0.3)
        ax.plot([0, 1], [0, 1], color='k', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xticks(np.linspace(0, 1, 6))
        plt.yticks(np.linspace(0, 1, 6))
        ax.tick_params(axis='x', direction='in')
        ax.tick_params(axis='y', direction='in')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        ax.set_title("(c)", fontsize=10)

    def _plot_effect_size(self):
        colors = sns.color_palette()
        ax = self.fig.add_subplot(self.gs[0, 1])
        ss = ax.get_subplotspec()
        ax.set_axis_off()
        gs = gridspec.GridSpecFromSubplotSpec(3, 1, ss, height_ratios=[0.35, 0.6, 0.05])
        ax = self.fig.add_subplot(gs[1, 0])
        #gs = gridspec.GridSpecFromSubplotSpec(1, 1, ss, wspace=10, hspace=10)
        #ax = self.fig.add_subplot(gs[0, 0])
        # plot lines for small, medium, large effect
        ax.vlines([0.09, 0.09], -0.2, 0.2, color='0.5', linestyle='-', lw=1)
        ax.vlines([0.25, 0.25], -0.2, 0.2, color='0.5', linestyle='-', lw=1)
        ax.vlines([0.40, 0.40], -0.2, 0.2, color='0.5', linestyle='-', lw=1)
        ax.text(0.045, 0.15, "small", fontsize=9, horizontalalignment='center')
        ax.text(0.17, 0.15, "medium", fontsize=9, horizontalalignment='center')
        ax.text(0.325, 0.15, "large", fontsize=9, horizontalalignment='center')

        ax.plot([self.partial_eta_squared_lower, self.partial_eta_squared_upper], [0, 0], alpha=0.5, linewidth=3,
                solid_capstyle='round', color='k')
        ax.plot([0, self.partial_eta_squared], [0, 0], alpha=0.5, linewidth=9,
                 color=colors[0], solid_capstyle='butt')
        ax.set(yticklabels=[])
        ax.set(ylabel=None)
        ax.tick_params(bottom=False)
        sns.despine(ax=ax, bottom=True)
        plt.xlabel(r'$\eta^2_{partial}$')
        plt.xticks([0, 0.09, 0.25, 0.4])
        plt.ylim([-0.2, 0.2])
        plt.xlim([0, 0.45])
        ax.set_yticks([])
        ax.set_title("(b)", fontsize=10)

    def common_language_effect_size_statistic(self):
        df = self.df
        x = self.variable_name
        group = self.group_name
        group_names = df[group].unique()
        control = df[df[group] == group_names[0]]
        treatment = df[df[group] == group_names[1]]
        var_pooled = np.sqrt(np.var(control[x]) + np.var(treatment[x]))
        diff = np.mean(control[x]) - np.mean(treatment[x])
        z = (0 - np.abs(diff)) / var_pooled
        p = 1 - norm.cdf(z)
        self.CLESS = p


if __name__ == '__main__':
    df = pd.DataFrame({'Values': np.random.randn(1000), 'residuals': np.random.randn(1000),
                       'Groups': np.random.randint(1, 3, 1000)})
    df.loc[df['Groups'] == 1, 'Values'] = df.loc[df['Groups'] == 1, 'Values'] + 3
    df['Groups'] = df['Groups'].astype("category")

    cohens_u = 0.54
    partial_eta_squared = 0.01
    plotter = ClassificationEffectSizePlotMatplotlib(df=df,
                                                     residuals=df,
                                                     group_name='Groups',
                                                     variable_name='Values',
                                                     partial_eta_squared=0.05,
                                                     partial_eta_squared_upper=0.1)
    plotter.plot()
    plt.show()
