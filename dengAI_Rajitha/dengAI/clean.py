
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('datasets/dengue_features_train.csv', parse_dates=[3])

target = pd.read_csv('datasets/dengue_labels_train.csv')

df_sj = df[df['city'] == 'sj']
df_iq = df[df['city'] == 'iq']

df_sj = df_sj.fillna(method = 'ffill')
df_iq = df_iq.fillna(method = 'ffill')

# df = target.set_index(['year', 'weekofyear'])
