import numpy as np


class SampleFilter:
    FILTERS = ['filter_all', 'filter_hc_mdd', 'filter_hc_mdd_acute', 'filter_hc_mdd_remitted',
               'filter_hc_mdd_moderate', 'filter_hc_mdd_severe', 'filter_hc_mdd_extreme20',
               'filter_hc_mdd_gaf_20', 'filter_hc_mdd_gaf_50', 'filter_hc_mdd_ctq_20',
               'filter_hc_mdd_ctq_50', 'filter_mdd_gaf_20', 'filter_mdd_gaf_50', 'filter_mdd_ctq_20',
               'filter_mdd_ctq_50', 'filter_hc_mdd_pss_20',
               'filter_hc_mdd_bodily_pain_25', 'filter_hc_mdd_general_health_25',
               'filter_hc_mdd_physical_functioning_25', 'filter_hc_mdd_vitality_25',
               'filter_hc_mdd_social_functioning_25', 'filter_hc_mdd_mental_health_25',
               'filter_hc_mdd_female', 'filter_hc_mdd_male',
               'filter_hc_mdd_acute_male', 'filter_hc_mdd_acute_female',
               'filter_hc_mdd_severe_male', 'filter_hc_mdd_severe_female',
               'filter_hc_mdd_marburg', 'filter_hc_mdd_muenster',
               'filter_hc_mdd_acute_marburg', 'filter_hc_mdd_acute_muenster',
               'filter_hc_mdd_severe_marburg', 'filter_hc_mdd_severe_muenster',

               'filter_mdd_bd',
               'filter_mdd_bd_female', 'filter_mdd_bd_male',
               'filter_mdd_bd_marburg', 'filter_mdd_bd_muenster',

               'filter_hc_bd',
               'filter_hc_bd_female', 'filter_hc_bd_male',
               'filter_hc_bd_marburg', 'filter_hc_bd_muenster',
               ]

    def apply_filter(self, filter_name, df):
        if filter_name not in self.FILTERS:
            raise NotImplementedError('Filter method not available. Available methods are: {}'.format(self.FILTERS))
        else:
            func = getattr(self, filter_name)
            df['Group'] = df['Group'].replace({1: 'HC', 2: 'MDD',
                                               3: 'BD', 4: 'SA',
                                               5: 'SZ'})
            return func(df)

    @staticmethod
    def filter_all(df):
        return df, [True] * df.shape[0]

    @staticmethod
    def filter_hc_mdd(df):
        filter_var = df['Group'].isin(['HC', 'MDD'])
        return df[filter_var], filter_var

    @staticmethod
    def filter_mdd_bd(df):
        filter_var = df['Group'].isin(['BD', 'MDD'])
        return df[filter_var], filter_var

    @staticmethod
    def filter_hc_bd(df):
        filter_var = df['Group'].isin(['BD', 'HC'])
        return df[filter_var], filter_var

    @staticmethod
    def filter_hc_mdd_acute(df):
        hc_filter = df['Group'].isin(['HC'])
        mdd_filter = df['Group'].isin(['MDD'])

        # find remission
        acute_filter = df['Rem'] < 2
        acute_mdd_filter = mdd_filter & acute_filter

        filter_var = hc_filter | acute_mdd_filter
        return df[filter_var], filter_var

    @staticmethod
    def filter_hc_mdd_remitted(df):
        hc_filter = df['Group'].isin(['HC'])
        mdd_filter = df['Group'].isin(['MDD'])

        # find remission
        rem_filter = df['Rem'] == 2
        rem_mdd_filter = mdd_filter & rem_filter

        filter_var = hc_filter | rem_mdd_filter
        return df[filter_var], filter_var

    @staticmethod
    def filter_hc_mdd_moderate(df):
        hc_filter = df['Group'].isin(['HC'])
        mdd_filter = df['Group'].isin(['MDD'])

        # find hosp
        hosp_filter = df['Hosp'] > 0
        hosp_mdd_filter = mdd_filter & hosp_filter

        filter_var = hc_filter | hosp_mdd_filter
        return df[filter_var], filter_var

    @staticmethod
    def filter_hc_mdd_severe(df):
        hc_filter = df['Group'].isin(['HC'])
        mdd_filter = df['Group'].isin(['MDD'])

        # find hosp
        hosp_filter = df['Hosp'] > 1
        hosp_mdd_filter = mdd_filter & hosp_filter

        filter_var = hc_filter | hosp_mdd_filter
        return df[filter_var], filter_var

    @staticmethod
    def filter_hc_mdd_extreme20(df):
        hc_filter = df['Group'].isin(['HC'])
        mdd_filter = df['Group'].isin(['MDD'])

        # find percentile
        dep_ep_cutoff = np.percentile(df.loc[mdd_filter, 'DepEp'].dropna().values, 80)
        extreme_filter = df['DepEp'] >= dep_ep_cutoff
        extreme_mdd_filter = mdd_filter & extreme_filter

        filter_var = hc_filter | extreme_mdd_filter
        return df[filter_var], filter_var

    @staticmethod
    def filter_by_variable(df, variable_name, groups: list = ['HC', 'MDD'], percentile: int = 50):
        group_filter = df['Group'].isin(groups)

        # find percentile
        cutoff_low = np.percentile(df.loc[group_filter, variable_name].dropna().values, percentile)
        cutoff_high = np.percentile(df.loc[group_filter, variable_name].dropna().values, 100 - percentile)
        filter_low = df[variable_name] >= cutoff_high
        filter_high = df[variable_name] <= cutoff_low

        # filter
        df['Group'] = np.zeros(df.shape[0])
        df.loc[filter_low, 'Group'] = "Low_" + variable_name
        df.loc[filter_high, 'Group'] = "High_" + variable_name

        filter_var = group_filter & (filter_low | filter_high)
        return df[filter_var], filter_var

    @staticmethod
    def filter_hc_mdd_gaf_50(df):
        return SampleFilter.filter_by_variable(df, 'GAFscore', ['HC', 'MDD'], 50)

    @staticmethod
    def filter_hc_mdd_gaf_20(df):
        return SampleFilter.filter_by_variable(df, 'GAFscore', ['HC', 'MDD'], 20)

    @staticmethod
    def filter_hc_mdd_ctq_50(df):
        return SampleFilter.filter_by_variable(df, 'CTQ_Sum', ['HC', 'MDD'], 50)

    @staticmethod
    def filter_hc_mdd_ctq_20(df):
        return SampleFilter.filter_by_variable(df, 'CTQ_Sum', ['HC', 'MDD'], 20)

    @staticmethod
    def filter_mdd_gaf_50(df):
        return SampleFilter.filter_by_variable(df, 'GAFscore', ['MDD'], 50)

    @staticmethod
    def filter_mdd_gaf_20(df):
        return SampleFilter.filter_by_variable(df, 'GAFscore', ['MDD'], 20)

    @staticmethod
    def filter_mdd_ctq_50(df):
        return SampleFilter.filter_by_variable(df, 'CTQ_Sum', ['MDD'], 50)

    @staticmethod
    def filter_mdd_ctq_20(df):
        return SampleFilter.filter_by_variable(df, 'CTQ_Sum', ['MDD'], 20)

    @staticmethod
    def filter_hc_mdd_pss_20(df):
        return SampleFilter.filter_by_variable(df, 'PSS_Sum', ['HC', 'MDD'], 20)

    @staticmethod
    def filter_hc_mdd_mental_health_25(df):
        return SampleFilter.filter_by_variable(df, 'SF36_MentalHealth', ['HC', 'MDD'], 25)

    @staticmethod
    def filter_hc_mdd_social_functioning_25(df):
        return SampleFilter.filter_by_variable(df, 'SF36_SocialFunctioning', ['HC', 'MDD'], 25)

    @staticmethod
    def filter_hc_mdd_physical_functioning_25(df):
        return SampleFilter.filter_by_variable(df, 'SF36_PhysicalFunctioning', ['HC', 'MDD'], 25)

    @staticmethod
    def filter_hc_mdd_general_health_25(df):
        return SampleFilter.filter_by_variable(df, 'SF36_GeneralHealth', ['HC', 'MDD'], 25)

    @staticmethod
    def filter_hc_mdd_vitality_25(df):
        return SampleFilter.filter_by_variable(df, 'SF36_Vitality', ['HC', 'MDD'], 25)

    @staticmethod
    def filter_hc_mdd_bodily_pain_25(df):
        return SampleFilter.filter_by_variable(df, 'SF36_BodilyPain', ['HC', 'MDD'], 25)

    @staticmethod
    def filter_hc_mdd_female(df):
        group_filter = df['Group'].isin(['HC', 'MDD'])

        # select only females
        gender_filter = df['Geschlecht'] == 2

        filter_var = group_filter & gender_filter
        return df[filter_var], filter_var

    @staticmethod
    def filter_hc_mdd_male(df):
        group_filter = df['Group'].isin(['HC', 'MDD'])

        # select only males
        gender_filter = df['Geschlecht'] == 1

        filter_var = group_filter & gender_filter
        return df[filter_var], filter_var

    @staticmethod
    def filter_mdd_bd_female(df):
        group_filter = df['Group'].isin(['BD', 'MDD'])

        # select only females
        gender_filter = df['Geschlecht'] == 2

        filter_var = group_filter & gender_filter
        return df[filter_var], filter_var

    @staticmethod
    def filter_mdd_bd_male(df):
        group_filter = df['Group'].isin(['BD', 'MDD'])

        # select only males
        gender_filter = df['Geschlecht'] == 1

        filter_var = group_filter & gender_filter
        return df[filter_var], filter_var

    @staticmethod
    def filter_hc_bd_female(df):
        group_filter = df['Group'].isin(['BD', 'HC'])

        # select only females
        gender_filter = df['Geschlecht'] == 2

        filter_var = group_filter & gender_filter
        return df[filter_var], filter_var

    @staticmethod
    def filter_hc_bd_male(df):
        group_filter = df['Group'].isin(['BD', 'HC'])

        # select only males
        gender_filter = df['Geschlecht'] == 1

        filter_var = group_filter & gender_filter
        return df[filter_var], filter_var

    @staticmethod
    def filter_hc_mdd_acute_female(df):
        hc_filter = df['Group'].isin(['HC'])
        mdd_filter = df['Group'].isin(['MDD'])

        # select only females
        gender_filter = df['Geschlecht'] == 2

        acute_filter = df['Rem'] < 2
        acute_mdd_filter = mdd_filter & acute_filter

        filter_var = (hc_filter | acute_mdd_filter) & gender_filter

        return df[filter_var], filter_var

    @staticmethod
    def filter_hc_mdd_acute_male(df):
        hc_filter = df['Group'].isin(['HC'])
        mdd_filter = df['Group'].isin(['MDD'])

        # select only males
        gender_filter = df['Geschlecht'] == 1

        acute_filter = df['Rem'] < 2
        acute_mdd_filter = mdd_filter & acute_filter

        filter_var = (hc_filter | acute_mdd_filter) & gender_filter

        return df[filter_var], filter_var

    @staticmethod
    def filter_hc_mdd_severe_female(df):
        hc_filter = df['Group'].isin(['HC'])
        mdd_filter = df['Group'].isin(['MDD'])

        # select only females
        gender_filter = df['Geschlecht'] == 2

        hosp_filter = df['Hosp'] > 1
        hosp_mdd_filter = mdd_filter & hosp_filter

        filter_var = (hc_filter | hosp_mdd_filter) & gender_filter

        return df[filter_var], filter_var

    @staticmethod
    def filter_hc_mdd_severe_male(df):
        hc_filter = df['Group'].isin(['HC'])
        mdd_filter = df['Group'].isin(['MDD'])

        # select only males
        gender_filter = df['Geschlecht'] == 1

        hosp_filter = df['Hosp'] > 1
        hosp_mdd_filter = mdd_filter & hosp_filter

        filter_var = (hc_filter | hosp_mdd_filter) & gender_filter
        return df[filter_var], filter_var

    @staticmethod
    def filter_hc_mdd_marburg(df):
        group_filter = df['Group'].isin(['HC', 'MDD'])

        # select site
        site_filter = df['Site'] == 1

        filter_var = group_filter & site_filter
        return df[filter_var], filter_var

    @staticmethod
    def filter_hc_mdd_muenster(df):
        group_filter = df['Group'].isin(['HC', 'MDD'])

        # select site
        site_filter = df['Site'] == 2

        filter_var = group_filter & site_filter
        return df[filter_var], filter_var

    @staticmethod
    def filter_mdd_bd_marburg(df):
        group_filter = df['Group'].isin(['BD', 'MDD'])

        # select site
        site_filter = df['Site'] == 1

        filter_var = group_filter & site_filter
        return df[filter_var], filter_var

    @staticmethod
    def filter_mdd_bd_muenster(df):
        group_filter = df['Group'].isin(['BD', 'MDD'])

        # select site
        site_filter = df['Site'] == 2

        filter_var = group_filter & site_filter
        return df[filter_var], filter_var

    @staticmethod
    def filter_hc_bd_marburg(df):
        group_filter = df['Group'].isin(['HC', 'BD'])

        # select site
        site_filter = df['Site'] == 1

        filter_var = group_filter & site_filter
        return df[filter_var], filter_var

    @staticmethod
    def filter_hc_bd_muenster(df):
        group_filter = df['Group'].isin(['HC', 'BD'])

        # select site
        site_filter = df['Site'] == 2

        filter_var = group_filter & site_filter
        return df[filter_var], filter_var

    @staticmethod
    def filter_hc_mdd_acute_marburg(df):
        hc_filter = df['Group'].isin(['HC'])
        mdd_filter = df['Group'].isin(['MDD'])

        # select site
        site_filter = df['Site'] == 1

        acute_filter = df['Rem'] < 2
        acute_mdd_filter = mdd_filter & acute_filter

        filter_var = (hc_filter | acute_mdd_filter) & site_filter

        return df[filter_var], filter_var

    @staticmethod
    def filter_hc_mdd_acute_muenster(df):
        hc_filter = df['Group'].isin(['HC'])
        mdd_filter = df['Group'].isin(['MDD'])

        # select site
        site_filter = df['Site'] == 2

        acute_filter = df['Rem'] < 2
        acute_mdd_filter = mdd_filter & acute_filter

        filter_var = (hc_filter | acute_mdd_filter) & site_filter

        return df[filter_var], filter_var

    @staticmethod
    def filter_hc_mdd_severe_marburg(df):
        hc_filter = df['Group'].isin(['HC'])
        mdd_filter = df['Group'].isin(['MDD'])

        # select site
        site_filter = df['Site'] == 1

        hosp_filter = df['Hosp'] > 1
        hosp_mdd_filter = mdd_filter & hosp_filter

        filter_var = (hc_filter | hosp_mdd_filter) & site_filter

        return df[filter_var], filter_var

    @staticmethod
    def filter_hc_mdd_severe_muenster(df):
        hc_filter = df['Group'].isin(['HC'])
        mdd_filter = df['Group'].isin(['MDD'])

        # select site
        site_filter = df['Site'] == 2

        hosp_filter = df['Hosp'] > 1
        hosp_mdd_filter = mdd_filter & hosp_filter

        filter_var = (hc_filter | hosp_mdd_filter) & site_filter
        return df[filter_var], filter_var