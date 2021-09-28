import os
import numpy as np
import pandas as pd


def summarize(results_folders: list,
              sample_filters: list,
              output_folder: str):
    """
    summarize results from multiple analyses
    creates table with modalities as rows and sample as columns
    table cells contain effect size or p value (can be chosen by user)
    :param folders:
    :return:
    """
    df_es = pd.DataFrame(columns=sample_filters)
    df_bacc = pd.DataFrame(columns=sample_filters)
    df_p = pd.DataFrame(columns=sample_filters)

    for modality_folder in results_folders:
        mod_name = os.path.basename(modality_folder)
        print("#" * 20)
        print("{}".format(mod_name))
        print("#" * 20)
        eff = list()
        bacc = list()
        p = list()

        for sample_filter in sample_filters:
            print("{}".format(sample_filter))
            folder = os.path.join(modality_folder, sample_filter)

            eff_file = os.path.join(folder, 'effect_size_results.csv')
            aov_file = os.path.join(folder, 'anova_results.csv')

            if os.path.exists(eff_file):
                eff_res = pd.read_csv(eff_file)
                eff.append(eff_res['Partial Eta2'].values[0])
                bacc.append(eff_res['BalancedAccuracy'].values[0])
            else:
                eff.append(np.nan)
                bacc.append(np.nan)

            if os.path.exists(aov_file):
                aov_res = pd.read_csv(aov_file)
                try:
                    p.append(aov_res.loc[1, 'p-corr'])
                except KeyError as ke:
                    p_corr = pd.read_csv(os.path.join(folder, 'tfce_p.csv'))['p-corr'][0]
                    p.append(p_corr)
            else:
                p.append(np.nan)

        df_es = df_es.append(pd.DataFrame([eff], index=[mod_name], columns=sample_filters))
        df_bacc = df_bacc.append(pd.DataFrame([bacc], index=[mod_name], columns=sample_filters))
        df_p = df_p.append(pd.DataFrame([p], index=[mod_name], columns=sample_filters))

    df_es.to_csv(os.path.join(output_folder, 'effect_size_summary.csv'))
    df_bacc.to_csv(os.path.join(output_folder, 'bacc_summary.csv'))
    df_p.to_csv(os.path.join(output_folder, 'p_summary.csv'))


if __name__ == "__main__":
    folders = ['../results/clinical/BDI',
               '../results/clinical/BigFive',
               '../results/clinical/CTQ',
               '../results/clinical/GAF',
               '../results/clinical/Neuropsychology',
               '../results/clinical/SF36',
               '../results/clinical/SocialSupport',
               '../results/freesurfer/Freesurfer',
               '../results/prs/Polygenic Risk Scores',
               '../results/hariri/Hariri Faces',
               '../results/dti/DTI Network Parameter',
               '../results/dti/DTI FA',
               '../results/dti/DTI MD',
               '../results/resting_state/RS Connectivities',
               '../results/rs/RS Network Parameter',
               ]
    filters = ['hc_mdd', 'hc_mdd_remitted', 'hc_mdd_acute', 'hc_mdd_severe', 'hc_mdd_extreme20']
    #summarize(folders, filters, './')

    from pipeline.steps import SummaryPlot
    import plotly
    plotly.io.orca.config.executable = '/opt/anaconda3/bin/orca'
    plotly.io.orca.config.use_xvfb = False
    for modality in folders:
        analyses = list()
        for filter_name in filters:
            analyses.append(os.path.join(modality, filter_name))

        SummaryPlot(name='summary_plot', pipeline=None, result_dir=modality)._execute(group=filters, result_folder_analyses=analyses)
