"""
Script to remove relatives when computing MDS components in a genetic analysis
"""
import os
import pandas as pd
import subprocess
import shutil

from pipeline.steps import PipelineStep
from pipeline.steps import SampleFilter
from multi_modality_datacreator.data_creator import run_prs


class PRSPreprocessing(PipelineStep):

    def _execute(self, subjects: str = '../raw_data/prs/PRS_DataFreeze1-3-30102020_macbook.csv',
                 prs_file: str = '../raw_data/prs/PRS_DataFreeze1-3-30102020_macbook.csv',
                 pheno_file: str = '../raw_data/prs/PRS_DataFreeze1-3-30102020_macbook.csv',
                 output_path: str = '../data/prs/',
                 plink_path: str = '/spm-data/vault-data3/mmll/data/FOR2107/session01/genetics/Sciebo',
                 *args, **kwargs):
        self.output_path = output_path

        # do basic preprocessing
        run_prs(sample_file=subjects, prs_file=prs_file, output_path=output_path)

        # load prs data
        df = pd.read_csv(os.path.join(output_path, 'prs_complete.csv'), index_col='Proband')

        # load phenotype data
        df_pheno = pd.read_csv(pheno_file, na_values=[-99], sep=';', decimal=',', index_col='Proband')
        df = df.join(df_pheno, how='inner')

        # filter subjects
        df, filter_variable = SampleFilter().apply_filter(self.pipeline.filter_name, df)

        # write sample information to file
        n_full = df.shape[0]
        self.fid.write('N full sample: {}\n'.format(n_full))

        # write subject_IDs including FID and IID
        sample_file = os.path.abspath(os.path.join(output_path, 'prs_initial_sample.txt'))
        subject_ids = df[["FID", "IID"]]
        subject_ids.to_csv(sample_file, sep=" ", index=None, header=None)

        # remove relatives
        self._find_relatives(plink_path=plink_path,
                             sample_file=sample_file)
        new_sample_file = self._remove_relatives(plink_path=plink_path,
                                                 sample_file=sample_file)

        # calculate MDS components
        self._calculate_mds(plink_path=plink_path,
                            sample_file=new_sample_file)

        # reduce original dataframe to sample without relatives and add MDS components
        df['Proband'] = df.index
        df.set_index('FID', inplace=True)
        mds = pd.read_table(os.path.join(self.output_path, 'FOR2107_selection_mds.txt'), sep=" ", index_col='FID')
        df = df.join(mds, how='right', rsuffix='r')
        df['FID'] = df.index
        df.set_index('Proband', inplace=True)
        self.fid.write('N sample after removing relatives: {}\n'.format(df.shape[0]))

        # write file
        df.to_csv(os.path.join(output_path, 'prs_complete_norel.csv'))

    @staticmethod
    def _find_relatives(plink_path, sample_file):
        subprocess.run("./calc_relatives.sh FOR2107 {}".format(sample_file),
                       shell=True, check=True, cwd=plink_path)
        return

    @staticmethod
    def _remove_relatives(plink_path: str, sample_file: str):
        sample = pd.read_table(sample_file, names=["FID", "IID"], header=None, sep=" ")
        relatives = pd.read_table(os.path.join(plink_path, 'FOR2107_selection_genome.txt'), sep=" ")

        # print some info
        print("Removing relatives...\n")
        print("Sample N: {}\n".format(sample.shape[0]))

        # find subjects that are contained in list more than once
        unique_fid1 = set(relatives["FID1"])
        unique_fid2 = set(relatives["FID2"])

        # remove lowest number of subjects possible
        if len(unique_fid1) >= len(unique_fid2):
            subs_to_remove = unique_fid2
        else:
            subs_to_remove = unique_fid1

        print("Removing {} subjects.../n".format(len(subs_to_remove)))
        reduced_sample = sample[~sample["FID"].isin(subs_to_remove)]
        filename = sample_file[:-4] + "_norel.txt"
        reduced_sample.to_csv(filename, sep="\t", index=None, header=None)
        print("Writing file.../n")
        return filename

    def _calculate_mds(self, plink_path, sample_file):
        subprocess.run("./calc_mds.sh FOR2107 {}".format(sample_file),
                       shell=True, check=True, cwd=plink_path)

        # copy MDS components to output folder
        shutil.copy(os.path.join(plink_path, 'FOR2107_selection_mds.txt'),
                    os.path.join(self.output_path, 'FOR2107_selection_mds.txt'))
