import os
import numpy as np
from nilearn.image import load_img
import pandas as pd

from pipeline.steps import PipelineStep
from pipeline.steps.statistics.spm_glm import SPMFullFactorial


class SpmGlmStep(PipelineStep):

    def _execute(self, overwrite: bool, matlab_path: str, spm_path: str, tfce_perms: int = 0,
                 correction_method: str = 'RandomField',
                 type: str = 'structural', group_variable: str = None, load_existing: bool = False,
                 *args, **kwargs):
        peak_file = os.path.join(self.pipeline.result_dir, 'peak_voxel_data.csv')
        self.tfce_perms = tfce_perms
        self.correction_method = correction_method

        if not load_existing:
            model = SPMFullFactorial(result_dir=self.result_dir,
                                     df=self.pipeline.df,
                                     spm_path=spm_path,
                                     matlab_path=matlab_path,
                                     type=type,
                                     overwrite=overwrite)
            if group_variable is None:
                model.add_factor('Group', self.pipeline.X_names[0])
            else:
                model.add_factor(group_variable, self.pipeline.X_names[0])
            model.add_continuous_covariate('Alter')
            model.add_categorical_covariate('Geschlecht')
            model.add_continuous_covariate('Dummy_BC_MR_pre')
            model.add_continuous_covariate('Dummy_BC_MR_post')
            if type == 'structural':
                model.add_continuous_covariate('TIV')

            if tfce_perms:
                model.add_tfce(tfce_perms)
            model.estimate()

            peak_ind = self.find_peak_voxel()
            self.pipeline.df['peak_voxel'] = self.extract_voxel_data(peak_ind)
            self.pipeline.X_names = ['peak_voxel']
            self.pipeline.df.to_csv(peak_file)
        else:
            self.pipeline.df = pd.read_csv(peak_file, index_col='Proband')
            self.pipeline.X_names = ['peak_voxel']

        pd.DataFrame({'X_best': 0}, index=[0]).to_csv(os.path.join(self.result_dir, 'X_best.csv'))

    def find_peak_voxel(self):
        # this I checked with the SPM graphical user interface, the contrast of interest (group differences) is the
        # second default SPM contrast, that's why I load spmF_0002.nii
        fmap_img = load_img(os.path.join(self.result_dir, 'SPM', 'spmF_0002.nii'))
        img_data = fmap_img.get_data()
        if self.correction_method == 'TFCE':
            pmap_img = load_img(os.path.join(self.result_dir, 'SPM', 'TFCE_log_pFDR_0002.nii'))
        elif self.correction_method == 'RandomField':
            pmap_img = load_img(os.path.join(self.result_dir, 'SPM', 'F_log_pFWE_0002.nii'))
        else:
            raise NotImplementedError("Method for correcting multiple comparisons has to be one of the following: "
                                      "'RandomField' or 'TFCE'")
        img_data_p = pmap_img.get_data()
        max_log_p = np.nanargmax(img_data_p, axis=None)
        ind = np.unravel_index(max_log_p, img_data.shape)
        p = np.power(10, -img_data_p[ind])
        print("Min p value found for voxel {}: p-Value {} F-Value {}".format(ind, p, img_data[ind]))
        pd.DataFrame({'p-corr': p}, index=[0]).to_csv(os.path.join(self.result_dir, 'spm_p.csv'))
        return ind

    def extract_voxel_data(self, ind):
        filenames = self.pipeline.df[self.pipeline.X_names[0]]
        data = []
        print("Extracting voxel data from")
        for file in filenames:
            fmap_img = load_img(file)
            img_data = fmap_img.get_data()
            data.append(img_data[ind])
        return np.asarray(data)
