import os
import pandas as pd

from photonai.base import Hyperpipe, PipelineElement, OutputSettings, Switch
from sklearn.model_selection import StratifiedKFold, ShuffleSplit


def run_pipeline():

    results_dir = '/scratch/tmp/e0trap/projects/multi_modality/results/freesurfer/'
    data_dir = '/scratch/tmp/e0trap/projects/multi_modality/data/freesurfer/'
    df = pd.read_csv(os.path.join(data_dir, 'cleaned_data.csv'))
    X_names = pd.read_csv(os.path.join(data_dir, 'X_names.csv'))
    X = df.loc[:, X_names['X_names']]
    y = df['Group']

    """ define hyperpipe """
    pipe = Hyperpipe('basic_pipe',
                     output_settings=OutputSettings(project_folder=results_dir),
                     optimizer='grid_search',
                     metrics=['balanced_accuracy', 'sensitivity', 'specificity'],
                     best_config_metric='balanced_accuracy',
                     outer_cv=StratifiedKFold(n_splits=5, random_state=42),
                     inner_cv=ShuffleSplit(test_size=0.2, n_splits=5, random_state=42),
                     nr_of_processes=2,
                     cache_folder='/scratch/tmp/e0trap/projects/multi_modality/cache',
                     verbosity=1
                     )

    """ add pca """
    pipe += PipelineElement('PCA', hyperparameters={'n_components': None}, test_disabled=True)

    """ add transformer elements """
    transformer_switch = Switch('TransformerSwitch')
    transformer_switch += PipelineElement('LassoFeatureSelection',
                                          hyperparameters={'percentile_to_keep': [0.2],
                                                           'alpha': 1}, test_disabled=True)
    transformer_switch += PipelineElement('FClassifSelectPercentile',
                                          hyperparameters={'percentile': [20]})

    """ add estimator elements """
    estimator_switch = Switch('EstimatorSwitch')
    estimator_switch += PipelineElement('LinearSVC')
    estimator_switch += PipelineElement('SVC', kernel='rbf')
    estimator_switch += PipelineElement('RandomForestClassifier')
    estimator_switch += PipelineElement('AdaBoostClassifier')
    estimator_switch += PipelineElement('LogisticRegression')

    pipe += transformer_switch
    pipe += estimator_switch

    """ fit hyperpipe """
    pipe.fit(X, y)


""" allows to run pipeline.py outside of __main__.py """
if __name__ == '__main__':
    run_pipeline()