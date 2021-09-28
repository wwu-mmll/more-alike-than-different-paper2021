import pandas as pd
import numpy as np


df = pd.read_csv('../data/cat12/cat12_complete.csv', index_col='Proband',
                         na_values=[-99, 'nan'])

df['Group'] = df['Group'].replace({1: 'HC', 2: 'MDD'})
df = df[df['Group'].isin(['HC', 'MDD'])]
df.dropna(subset=['filename_mwp1'], inplace=True)

df = df[['Group', 'Alter', 'Geschlecht']]
#df.groupby('Group')
print(df.describe())
print(df['Group'].value_counts())
print(df.groupby('Group')['Geschlecht'].value_counts())


debug = True