import os
import pandas as pd
import numpy as np

from externals.multi_modality_creator.multi_modality_datacreator.data_merger import DtiPrep, RsPrep
from pipeline.steps import PipelineStep, SampleFilter


class RestingStateData(PipelineStep):

    def _execute(self, sample_file: str, data_path: str, atlas: str = 'Schaefer100-17', *args, **kwargs):
        df = pd.read_csv(sample_file, index_col='Proband', na_values=[-99, 'nan', -99.0], decimal=',', sep=';')

        # load dataframe that has been created in preprocessing step
        rs_valid_sample = pd.read_csv(os.path.join(data_path, 'rs_complete.csv'))[['Proband']]
        rs_valid_sample.set_index('Proband', inplace=True)
        df = df.merge(rs_valid_sample, how='inner', on='Proband')

        df, _ = SampleFilter().apply_filter(self.pipeline.filter_name, df)

        X, rois = RsPrep.prepare(proband_list=df.index.tolist(),
                                 output_path=data_path,
                                 atlas=atlas,
                                 save_numpy=False,
                                 return_roi_names=True)
        rois = ["region_{}".format(i) for i in range(100)]

        # get connectivity names
        # create matrix with connectivity names (roi with roi) and apply triu_indices
        n_rois = len(rois)
        roi_matrix = np.empty((n_rois, n_rois), dtype=object)
        for a_i, roi_a in enumerate(rois):
            for b_i, roi_b in enumerate(rois):
                roi_matrix[a_i, b_i] = roi_a.replace('-', '_') + '__' + roi_b.replace('-', '_')

        triu_ind = np.triu_indices(n_rois, 1)
        X_names = roi_matrix[triu_ind]

        # merge X into dataframe
        df_X = pd.DataFrame(X, columns=list(X_names), index=df.index)
        df = pd.concat([df, df_X], axis=1)

        self.pipeline.X_names = list(X_names)
        self.pipeline.df = df
        self.pipeline.X = X


class DTIData(PipelineStep):

    def _execute(self, sample_file: str, data_path: str, parameter: str = None,
                 percentage_edge_present: float = 0.95, *args, **kwargs):

        df = pd.read_csv(sample_file, index_col='Proband', na_values=[-99, 'nan', -99.0], decimal=',', sep=';')

        # load dataframe that has been created in preprocessing step
        rs_valid_sample = pd.read_csv(os.path.join(data_path, 'dti_complete.csv'))[['Proband']]
        rs_valid_sample.set_index('Proband', inplace=True)
        df = df.merge(rs_valid_sample, how='inner', on='Proband')

        df, _ = SampleFilter().apply_filter(self.pipeline.filter_name, df)

        X, rois = DtiPrep.prepare(proband_list=df.index.tolist(),
                                  output_path=data_path,
                                  parameter=parameter,
                                  save_numpy=False,
                                  return_roi_names=True)

        # check overlap
        overlap = np.sum(X != 0, axis=0)

        # use only regions for which all subjects have a connectivity value
        overlap_filter = overlap >= round((X.shape[0] * percentage_edge_present))

        # get connectivity names
        # create matrix with connectivity names (roi with roi) and apply triu_indices
        n_rois = len(rois)
        roi_matrix = np.empty((n_rois, n_rois), dtype=object)
        for a_i, roi_a in enumerate(rois):
            for b_i, roi_b in enumerate(rois):
                roi_matrix[a_i, b_i] = roi_a.replace('-', '_') + '__' + roi_b.replace('-', '_')

        triu_ind = np.triu_indices(n_rois, 1)
        X_names = roi_matrix[triu_ind]

        # merge X into dataframe
        X_overlap = X[:, overlap_filter]
        X_overlap[X_overlap == 0] = np.nan
        X_names = X_names[overlap_filter]

        df_X = pd.DataFrame(X_overlap, columns=list(X_names), index=df.index)
        df = pd.concat([df, df_X], axis=1)

        self.pipeline.X_names = list(X_names)
        self.pipeline.df = df
        self.pipeline.X = X_overlap
