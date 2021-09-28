import numpy as np
import matplotlib.pylab as plt
from tqdm import tqdm
import pandas as pd
import os
from statsmodels.stats.multitest import multipletests

from externals.gists.mclust.gaussian_mixture import Mclust

from pipeline.steps.statistics.anova import AnovaES
from pipeline.steps import PipelineStep


class LargestSampleEffectSize(PipelineStep):

    def _execute(self, group_contrast: str = None,
                 covariates: list = None, ss_type: int = 3,
                 minimum_effect_size: float = 0.20, *args, **kwargs):

        # try random data
        self.pipeline.X_names = ['random_1', 'random_2']
        self.pipeline.df[self.pipeline.X_names[0]] = np.random.randn(self.pipeline.df.shape[0])
        self.pipeline.df[self.pipeline.X_names[1]] = np.random.randn(self.pipeline.df.shape[0])

        df_mdd = self.pipeline.df[self.pipeline.df['Group'] == 'MDD']
        df_hc = self.pipeline.df[self.pipeline.df['Group'] == 'HC']
        sample_size_list = list(range(5, df_mdd.shape[0], 20))

        effect_sizes_all_variables = list()

        for x in tqdm(self.pipeline.X_names):
            anova = AnovaES(data=self.pipeline.df, group_contrast='C(Group, Sum)', covariates=covariates,
                            ss_type=ss_type)

            try:
                aov = anova.anova_es(x)
            except BaseException as e:
                print(e)
                continue

            resid = anova.calculate_residuals(x)

            # calculate mean for groups
            mean_hc = np.mean(resid[self.pipeline.df['Group'] == 'HC'])
            mean_mdd = np.mean(resid[self.pipeline.df['Group'] == 'MDD'])

            # sort MDDs
            mdd_order = np.argsort(resid[self.pipeline.df['Group'] == 'MDD'])
            if mean_hc < mean_mdd:
                mdd_order = mdd_order[::-1]

            eff_list = list()

            for sample_size in sample_size_list:
                # rerun ANOVA with reduced MDD sample
                reduced_mdd = df_mdd.iloc[mdd_order[:sample_size]]

                combined_df = pd.concat([df_hc, reduced_mdd])
                reduced_anova = AnovaES(data=combined_df, group_contrast='C(Group, Sum)', covariates=covariates,
                                        ss_type=ss_type)
                reduced_aov = reduced_anova.anova_es(x)
                eff_list.append(reduced_aov.loc[group_contrast, 'np2'])

            effect_sizes_all_variables.append(eff_list)

        eff = np.asarray(effect_sizes_all_variables)
        eff_thres = eff > minimum_effect_size
        indices_largest_effects = np.argwhere(eff_thres)
        # find variable with largest sample size and effect size larger than minimum_effect_size
        index_best_variable = np.max(indices_largest_effects, axis=0)[0]

        plt.figure()
        plt.plot(sample_size_list, eff[index_best_variable])
        plt.xlabel("N")
        plt.ylabel("Partial Eta2")
        plt.title(self.pipeline.X_names[index_best_variable])
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'largest_sample_largest_effect.png'))

        df_eff = pd.DataFrame({'PartialEta2': eff[index_best_variable]}, index=sample_size_list)
        df_eff.to_csv(os.path.join(self.result_dir, 'largest_sample_largest_effect.csv'))


class GaussianMixtureModel(PipelineStep):
    """
    Gaussian Mixture Model to find subpopulations within MDDs

    - for each variable run ANOVA first
    - extract residuals
    - fit Gaussian Mixture model on residuals using Mclust
    - run bootstrapped likelihood ratio test for one versus two Gaussians
    - correct for multiple comparisons across variables using FDR (Benjamini & Hochberg)
    """
    def _execute(self, group_contrast: str = None,
                 covariates: list = None, ss_type: int = 3, *args, **kwargs):

        p_values = list()
        residuals = list()

        for x in tqdm(self.pipeline.X_names[:40]):
            anova = AnovaES(data=self.pipeline.df, group_contrast='C(Group, Sum)', covariates=covariates,
                            ss_type=ss_type)

            try:
                anova.anova_es(x)
            except BaseException as e:
                print(e)
                residuals.append(None)
                continue

            resid = anova.calculate_residuals(x)

            resid_mdd = resid[self.pipeline.df['Group'] == 'MDD']
            residuals.append(resid_mdd)

        mclust = Mclust(model_name='V', n_gaussians=2)

        for resid_mdd in residuals:
            if resid_mdd is None:
                p_values.append(1)
            else:
                # select components for GaussianMixtureModel
                p = mclust.bootstrap_lrt(X=resid_mdd, max_g=2)
                p_values.append(p[0])

        # correct for multiple comparisons across variables
        reject, p_vals_corr, _, _ = multipletests(p_values, method='fdr_bh')

        for i, sign in enumerate(reject):
            if sign:
                mclust.fit(residuals[i])
                mclust.plot_cluster(residuals[i], filename=os.path.join(self.result_dir,
                                                                        '{}_hist.png'.format(self.pipeline.X_names[i])))


class ComputeClusterability(PipelineStep):
    def _execute(self, group_contrast: str = None,
                 covariates: list = None, ss_type: int = 3, *args, **kwargs):

        dip_p_values = list()
        silverman_p_values = list()
        residuals = list()

        for x in tqdm(self.pipeline.X_names):
            anova = AnovaES(data=self.pipeline.df, group_contrast='C(Group, Sum)', covariates=covariates,
                            ss_type=ss_type)
            try:
                anova.anova_es(x)
            except BaseException as e:
                print(e)
                residuals.append(None)
                continue

            resid = anova.calculate_residuals(x)

            resid_mdd = resid[self.pipeline.df['Group'] == 'MDD']
            residuals.append(resid_mdd)

        from externals.gists.mclust.clusterability import Clusterability
        clusterability = Clusterability()

        for resid_mdd in residuals:
            if resid_mdd is None:
                dip_p_values.append(1)
                silverman_p_values.append(1)
            else:
                p = clusterability.clusterabilitytest(X=resid_mdd, reduction='none')
                dip_p_values.append(p[0])
                silverman_p_values.append(p[1])

        # correct for multiple comparisons across variables
        dip_reject, dip_p_vals_corr, _, _ = multipletests(dip_p_values, method='fdr_bh')
        sil_reject, sil_p_vals_corr, _, _ = multipletests(silverman_p_values, method='fdr_bh')

        print(np.sum(dip_reject))
        print(np.sum(sil_reject))

        for i, single_reject in enumerate(sil_reject):
            if single_reject:
                for n_g in [2, 3, 4]:
                    mclust = Mclust(model_name='V', n_gaussians=n_g)
                    mclust.fit(residuals[i])
                    cluster_preds = mclust.predict(residuals[i])
                    means = list()
                    for n_g_i in range(n_g):
                        means.append(np.mean(residuals[i][cluster_preds == n_g_i+1]))
                    best_cluster = np.argmax(means)
                    cluster_mdds = self.pipeline.df[self.pipeline.df['Group'] == 'MDD'][cluster_preds == best_cluster + 1]

                    df = self.pipeline.df[self.pipeline.df['Group'] == 'HC'].append(cluster_mdds)

                    anova = AnovaES(data=df, group_contrast='C(Group, Sum)', covariates=covariates,
                                    ss_type=ss_type)

                    try:
                        aov = anova.anova_es(self.pipeline.X_names[i])
                        print('Anova results with max cluster: eta2 = {}'.format(aov['np2']['C(Group, Sum)']))
                    except:
                        print("anova failed")

