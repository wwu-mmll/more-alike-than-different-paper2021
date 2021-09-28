import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.api import stats
from statsmodels.formula.api import ols, logit
from statsmodels.stats.multitest import multipletests
from resample.bootstrap import confidence_interval

from pipeline.steps import PipelineStep


class Anova(PipelineStep):

    def _execute(self, method: str = None, group_contrast: str = None,
                 covariates: list = None, ss_type: int = 3,
                 n_bootstrap: int = 1000,
                 *args, **kwargs):
        # filter sample with respect to nan in covariates
        n_all = self.pipeline.df.shape[0]
        self.pipeline.df = self.pipeline.df.dropna(subset=covariates)
        self.fid.write("\n### ANOVA\n")
        self.fid.write("\nDropping {} subjects due to NaNs in covariates\n".format(n_all - self.pipeline.df.shape[0]))

        anova = AnovaES(data=self.pipeline.df, group_contrast=group_contrast, covariates=covariates,
                        ss_type=ss_type, n_bootstrap=n_bootstrap)

        if method == 'mass_anova':
            best_x, results = anova.mass_anova(self.pipeline.X_names)
        elif method == 'anova_bootstrapped_es':
            results = anova.anova_bootstrapped_es(x=self.pipeline.X_names[0], ci_method='bca',
                                                  strata=self.pipeline.df['Group'])
            best_x = [0]
        else:
            raise NotImplemented("Method has to be 'mass_anova' or 'anova_bootstrapped_es'.")

        self.pipeline.df = self.pipeline.df.dropna(subset=[self.pipeline.X_names[best_x[0]]])
        group = self.pipeline.df['Group']
        residuals = anova.calculate_residuals(self.pipeline.X_names[best_x[0]])
        residuals = pd.DataFrame({'residuals': residuals, 'Group': group})
        residuals.to_csv(os.path.join(self.result_dir, 'residuals.csv'))
        self.pipeline.X_best = best_x
        self.fid.write("\nN Subjects: {} (HC={}, MDD={}, BD={})\n".format(self.pipeline.df.shape[0],
                                                        self.pipeline.df[self.pipeline.df['Group'] == 'HC'].shape[0],
                                                        self.pipeline.df[self.pipeline.df['Group'] == 'MDD'].shape[0],
                                                        self.pipeline.df[self.pipeline.df['Group'] == 'BD'].shape[0]))
        self.fid.write("\nContrast: {}\n".format(group_contrast))
        self.fid.write("\nCovariates: {}\n".format(covariates))
        self.fid.write("\nSS Type: {}\n".format(ss_type))
        self.fid.write("\nN Bootstrap Samples: {}\n".format(n_bootstrap))
        self.fid.write("\nN target variables: {}\n".format(len(self.pipeline.X_names)))

        self.fid.write("\nBest Variable: {}\n\n".format(self.pipeline.X_names[best_x[0]]))
        self.fid.write("\nSignificant Variables ({}): \n\n".format(len(best_x)))
        for i in best_x:
            sig_var = self.pipeline.X_names[i]
            self.fid.write("- {}\n".format(sig_var))
        self.fid.write("\n")
        self.fid.write(results.to_markdown())

        results.to_csv(os.path.join(self.result_dir, 'anova_results.csv'))
        pd.DataFrame({'X_best': self.pipeline.X_best}).to_csv(os.path.join(self.result_dir, 'X_best.csv'))


class AnovaES:
    """
    Add description
    """
    def __init__(self, data: pd.DataFrame, group_contrast: str, covariates: list, ss_type: int = 3,
                 n_bootstrap: int = 1000):
        self.data = data
        self.ss_type = ss_type
        self.group_contrast = group_contrast
        formula = '{{}} ~ {}'.format(group_contrast)
        for cov in covariates:
            formula += ' + {}'.format(cov)
        self.formula_template = formula
        self.formula_template_only_covariates = '{} ~ '
        for i, cov in enumerate(covariates):
            if i == 0:
                self.formula_template_only_covariates += '{}'.format(cov)
            else:
                self.formula_template_only_covariates += ' + {}'.format(cov)
        self.formula = None
        self.n_bootstrap = n_bootstrap

    def anova_es(self, x: str):
        formula = self.formula_template.format(x)
        data = self.data.copy()
        data = data.dropna(subset=[x])
        return self._anova_es(data, formula, self.ss_type)

    @staticmethod
    def _anova_es(data, formula, ss_type):
        """
        Run anova-like linear model with bootstrapped effect size (partial eta2)
        :return:
        """
        # Fit using statsmodels
        lm = ols(formula, data=data).fit()
        aov = stats.anova_lm(lm, typ=ss_type)

        aov = aov.reset_index()
        aov = aov.rename(columns={'index': 'Source',
                                  'sum_sq': 'SS',
                                  'df': 'DF',
                                  'PR(>F)': 'p-unc'})
        aov.index = aov['Source']

        aov['MS'] = aov['SS'] / aov['DF']
        # calculate (partial) eta squared
        aov['n2'] = aov['SS'] / np.sum(aov['SS'])
        aov['np2'] = (aov['F'] * aov['DF']) / (aov['F'] * aov['DF'] + aov.loc['Residual', 'DF'])
        # another way of calculating np2 is:
        # aov['np2'] = aov['SS'] / (aov['SS'] + aov['SS']['Residual']
        # this produces exactly the same partial eta squared values
        return aov

    def _fit_anova_bes(self, permutation: list):
        # apply bootstrapped indices to create bootstrap sample
        perm_data = self.data.iloc[permutation]
        aov = self._anova_es(perm_data, self.formula, self.ss_type)
        return aov['np2']

    def anova_bootstrapped_es(self,
                              x: str,
                              cl: float = 0.95,
                              ci_method: str = 'bca',
                              strata: list = None):
        """
        Run anova-like linear model and compute partial eta2 as effect size measure
        including bootstrapped confidence intervals
        :return:
        """
        aov = self.anova_es(x)
        permutation = np.arange(self.data.shape[0])
        self.formula = self.formula_template.format(x)
        ci = confidence_interval(self._fit_anova_bes, permutation, cl=cl, ci_method=ci_method,
                                 **{'strata': strata, 'size': self.n_bootstrap})
        aov['np2_BCI_low'] = ci[0]
        aov['np2_BCI_high'] = ci[1]
        return aov

    def mass_anova(self,
                   x_names: list,
                   alpha: float = 0.05,
                   correction_method: str = 'fdr_by'):
        """

        :param x_names:
        :param alpha:
        :param correction_method:
        :return:
        """
        pvals = list()
        fvals = list()
        for x in tqdm(x_names):
            try:
                aov = self.anova_es(x)
                pvals.append(aov['p-unc'][self.group_contrast])
                fvals.append(aov['F'][self.group_contrast])
            except BaseException as e:
                print(e)
                pvals.append(1)
                fvals.append(0)

        # correct p values
        reject, pvals_corrected, _, _ = multipletests(pvals=pvals,
                                                      alpha=alpha,
                                                      method=correction_method)
        print("{} of {} tests are significant after correction".format(np.sum(reject), len(reject)))

        # find most significant and strongest effect
        #best = np.argmax(fvals)
        order = np.argsort(fvals)[::-1]
        if np.sum(reject):
            order = order[:np.sum(reject)]
        else:
            order = [order[0]]
        best = order[0]
        aov = self.anova_bootstrapped_es(x_names[best])
        aov['p-corr'] = np.full_like(aov.shape[0], np.nan, dtype=np.double)
        aov.loc[self.group_contrast, 'p-corr'] = pvals_corrected[best]
        return order, aov

    def calculate_residuals(self, variable_of_interest: str):
        """
        Return residuals of a model WITHOUT group variable of interest
        :return:
        """
        data = self.data.copy()
        data = data.dropna(subset=[variable_of_interest])
        lm = ols(self.formula_template_only_covariates.format(variable_of_interest), data=data).fit()
        return lm.resid


if __name__ == "__main__":
    df = pd.read_excel('test_data.xlsx', na_values=-99)
    df = df[df['Group'] < 3]
    form = '{} ~ C(Group, Sum) + Alter'
    anova = AnovaES(data=df, group_contrast='C(Group, Sum)', covariates=['Alter'])
    #results = anova.anova_bootstrapped_es(x='BDI_Sum', ci_method='bca', strata=None)
    #print(results)
    #results = anova.anova_bootstrapped_es(x='BDI_Sum', ci_method='bca', strata=df['Group'])
    #print(results)
    best_x, results = anova.mass_anova(['BDI_Sum', 'CTQ_Sum'])
    print(best_x)
    print(results)
    debug = True
