import pandas as pd
import numpy as np


df = pd.read_csv('../results/Freesurfer/hc_mdd/residuals.csv')
mean_mdd = np.mean(df['residuals'][df['Group'] == 'MDD'])

v = np.sum(mean_mdd > df['residuals'][df['Group'] == 'HC']) / np.sum(df['Group'] == 'HC')
print(v)