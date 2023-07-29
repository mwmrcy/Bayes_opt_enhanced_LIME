import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.utils import Bunch

df = pd.read_csv("data/ildp.csv")
            #df = df.fillna(method='ffill')
            #df['Class'] = np.where(df['Class'] == 'Yes', 0, 1)
df = df.fillna(method='ffill')
df['class'] = np.where(df['class'] == 'Yes', 0, 1)
feature_names = list(df.columns)
target_names = np.array(['yes', 'no'])

data = np.array(df.iloc[:, 0:9])
target = np.array(df['class'])

print(df.loc[:,"class"])
#self.data = Bunch(data=data, target=target,target_names=target_names, feature_names=feature_names)
