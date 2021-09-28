from nipype.interfaces import matlab

import os
import shutil
import pandas as pd
pd.set_option('display.max_colwidth', 999)


class SPMFullFactorial:

    def __init__(self, result_dir: str, df: pd.DataFrame, matlab_path: str, spm_path: str,
                 type: str = 'structural', overwrite: bool = False):
        """
        SPMFullFactorial estimates an SPM model including contrasts of interest.
        This is done by creating a MATLAB batch structure that is passed to spm_jobman().
        MATLAB code execution is done via NiPype's MATLAB interface.

        :param result_dir: specify a results directory
        :param df: this dataframe should contain nifti paths and additional subject information
        :param matlab_path: path to a matlab executable (compiled versions are not working)
        :param spm_path: path to your SPM installation (for TFCE, Cat12 and TFCE toolbox should be inside the SPM
                         toolbox folder)
        :param type: can be either 'structural' or 'functional', will change default threshold from 0.1 (Cat12)
                     to 0 (functional)
        :param overwrite: True/False, overwrite existing SPM results
        """
        print("Using MATLAB instance from: {}".format(matlab_path))
        print("Using SPM instance from: {}".format(spm_path))
        matlab.MatlabCommand.set_default_matlab_cmd(matlab_path)
        matlab.MatlabCommand.set_default_paths(spm_path)

        self.result_dir = result_dir
        self.df = df.copy()
        self.overwrite = overwrite
        self.type = type
        self.job = ""
        self.factor_exists = False
        self.covariate_counter = 1

        # create results folder if it does not exist
        self._initialize_results()

        # initialize job string
        self._initialize_job()

    def _initialize_results(self):
        """
        Handles existing or non-existing SPM folder. If overwrite is set to True, this will delete an existing
        SPM results folder.
        
        :return:
        """
        self.result_dir = os.path.join(self.result_dir, 'SPM')
        if os.path.exists(self.result_dir):
            if not self.overwrite:
                raise RuntimeError("SPM folder already exists and overwrite is set to False.")
            print("SPM results folder already exists. Deleting folder contents.")
            shutil.rmtree(self.result_dir)
        os.makedirs(self.result_dir)

    def _finalize_job(self):
        """
        Simply adds last two lines in MATLAB script which will run SPM. This is done prior to running estimate().

        :return:
        """
        self.job += """
            spm('defaults', 'FMRI');
            spm_jobman('run', matlabbatch);
            """

    def _check_factor_exists(self):
        """
        Check if factor has been defined. A factor needs to be defined before covariates can be added. This is due to
        the necessary sorting of the dataframe according to the factor levels (SPM will need everything in this order).

        :return:
        """
        if not self.factor_exists:
            raise RuntimeError("Factor has to be defined first.")

    def estimate(self):
        """
        Run SPM estimation step.

        :return:
        """
        self._check_factor_exists()
        self._finalize_job()
        mlab = matlab.MatlabCommand(mfile=True, single_comp_thread=False)
        mlab.inputs.script = self.job
        print("Running SPM Estimate.")
        out = mlab.run()
        print(out.runtime.stdout)
        print("Done running SPM estimate.")

    def add_factor(self, factor_name: str, scans_variable: str):
        """
        Add a between group factor to the SPM full factorial model.

        :param factor_name: str, name of factor that should be a column name in the dataframe
        :param scans_variable: str, name of variable that holds nifti paths, should be a column name in the dataframe
        :return:
        """
        print("SPM Config: Adding factor.")
        # check number of factor levels
        factor_levels = self.df[factor_name].unique()
        n_levels = len(factor_levels)
        print("   Found {} factor levels".format(n_levels))
        for i in range(n_levels):
            print("   {} is {}".format(i+1, factor_levels[i]))
            
        # sort df according to factor levels
        self.df.sort_values(by=factor_name, inplace=True)

        factor_string = """
            matlabbatch{{1}}.spm.stats.factorial_design.des.fd.fact.name = '{}';  
            matlabbatch{{1}}.spm.stats.factorial_design.des.fd.fact.levels = {};      
            """.format(factor_name, n_levels)

        for i in range(1, n_levels+1):
            scans = self.df.loc[self.df[factor_name] == factor_levels[i - 1], scans_variable]
            scans = "'" + scans + ",1'"
            factor_string += """
                matlabbatch{{1}}.spm.stats.factorial_design.des.fd.icell({}).levels = {};
                matlabbatch{{1}}.spm.stats.factorial_design.des.fd.icell({}).scans = {{
                {}
                }};
                """.format(i, i, i, scans.to_string(header=None, index=None))

        self.job += factor_string
        self.factor_exists = True

    def add_categorical_covariate(self, covariate_name: str):
        """
        Add a categorical covariate. Dummy coding (necessary for SPM) will be performed automatically.

        :param covariate_name: str, name of covariate, should be a column name in the dataframe
        :return:
        """
        print("SPM Config: Adding categorical covariate: {}".format(covariate_name))
        self._check_factor_exists()

        dummies = pd.get_dummies(self.df[covariate_name], drop_first=True, prefix=covariate_name)

        n_levels = len(dummies.columns) + 1
        print("   Found {} covariate levels. Creating dummy coding.".format(n_levels))

        for (level_name, level) in dummies.iteritems():
            self._add_covariate(level.tolist(), level_name)

    def add_continuous_covariate(self, covariate_name: str):
        """
        Add a continuous covariate.

        :param covariate_name: str, name of covariate, should be a column name in the dataframe
        :return:
        """
        print("SPM Config: Adding continuous covariate: {}".format(covariate_name))
        self._check_factor_exists()
        self._add_covariate(self.df[covariate_name].tolist(), covariate_name)

    def _add_covariate(self, covariate: list, covariate_name: str):
        i = self.covariate_counter
        self.job += """
            matlabbatch{{1}}.spm.stats.factorial_design.cov({}).c = {};
            matlabbatch{{1}}.spm.stats.factorial_design.cov({}).cname = '{}';
            matlabbatch{{1}}.spm.stats.factorial_design.cov({}).iCFI = 1;
            matlabbatch{{1}}.spm.stats.factorial_design.cov({}).iCC = 1;
            """.format(i, covariate, i, covariate_name, i, i)
        self.covariate_counter += 1

    def add_tfce(self, n_perm: int = 5000):
        """
        Perform TFCE-based correction for multiple comparisons.

        :param n_perm: int, number of permutations to be run for tfce
        :return:
        """
        print("SPM Config: Adding TFCE with {} permutations.".format(n_perm))
        self.job += "matlabbatch{{4}}.spm.tools.tfce_estimate.spmmat = {{'{}'}};\n".format(os.path.join(self.result_dir,
                                                                                                      'SPM.mat'))
        self.job += "matlabbatch{{4}}.spm.tools.tfce_estimate.conspec.n_perm = {};\n".format(n_perm)
        self.job += """
            matlabbatch{4}.spm.tools.tfce_estimate.mask = '';
            matlabbatch{4}.spm.tools.tfce_estimate.conspec.titlestr = 'group_contrast_results_tfce';
            matlabbatch{4}.spm.tools.tfce_estimate.conspec.contrasts = 2;
            matlabbatch{4}.spm.tools.tfce_estimate.nuisance_method = 2;
            matlabbatch{4}.spm.tools.tfce_estimate.tbss = 0;
            matlabbatch{4}.spm.tools.tfce_estimate.E_weight = 0.5;
            matlabbatch{4}.spm.tools.tfce_estimate.singlethreaded = 0;
            matlabbatch{4}.spm.tools.tfce_estimate.nproc = 16;
            """

    def _initialize_job(self):
        self.job += "results_dir = '{}';".format(self.result_dir)
        self.job += """
            matlabbatch{1}.spm.stats.factorial_design.dir = {results_dir};
            
            matlabbatch{1}.spm.stats.factorial_design.des.fd.fact.dept = 0;
            matlabbatch{1}.spm.stats.factorial_design.des.fd.fact.variance = 1;
            matlabbatch{1}.spm.stats.factorial_design.des.fd.fact.gmsca = 0;
            matlabbatch{1}.spm.stats.factorial_design.des.fd.fact.ancova = 0; 
            matlabbatch{1}.spm.stats.factorial_design.des.fd.contrasts = 1;
            
            matlabbatch{1}.spm.stats.factorial_design.multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});
            """
        if self.type == 'structural':
            self.job += "matlabbatch{1}.spm.stats.factorial_design.masking.tm.tma.athresh = 0.1;"
        elif self.type == 'functional':
            self.job += "matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;"
        else:
            raise NotImplemented("Type has to be either 'functional' or 'structural' but was {}".format(self.type))
        self.job += """
            matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;
            matlabbatch{1}.spm.stats.factorial_design.masking.em = {''};
            matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
            matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
            matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;
            
            matlabbatch{2}.spm.stats.fmri_est.spmmat(1) = {fullfile(results_dir, 'SPM.mat')};
            matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
            matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;
            
            matlabbatch{3}.spm.stats.con.spmmat(1) = {fullfile(results_dir, 'SPM.mat')};
            matlabbatch{3}.spm.stats.con.consess{1}.fcon.name = 'group_contrast';
            matlabbatch{3}.spm.stats.con.consess{1}.fcon.weights = [1 0];
            matlabbatch{3}.spm.stats.con.consess{1}.fcon.sessrep = 'none';
            matlabbatch{3}.spm.stats.con.delete = 0;        
            """


if __name__ == '__main__':

    # load example data
    data = pd.read_csv('../../../analyses/for2107_test_data.csv')

    # create full factorial model
    model = SPMFullFactorial(result_dir='./tmp/',
                             df=data,
                             spm_path='/spm-data/vault-data3/mmll/software/spm12_uncompiled/spm12',
                             matlab_path='/spm-data/it-share/public-applications/MATLAB/R2019a/bin/matlab',
                             overwrite=True)

    # add factor and covariates
    model.add_factor('Group', 'gray_matter_absolute_path')
    model.add_continuous_covariate('Alter')
    model.add_categorical_covariate('Geschlecht')

    # # optionally, add TFCE correction
    # model.add_tfce(n_perm=50)

    # run model
    model.estimate()
