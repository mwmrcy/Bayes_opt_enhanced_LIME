import glob
import pandas as pd
import numpy as np
from sklearn.utils import Bunch

# Get CSV files list from a folder
class LoadDataset:
    def __init__(self):
        
        dataset= pd.read_csv('data/GRF_data_to_use.csv')
        dataset = dataset.fillna(method='ffill')
       # dataset = dataset.iloc[:,1:]
        dataset['CLASS_LABEL']= np.where(dataset['CLASS_LABEL']=='UHC',0,1)
        feature_names = list(dataset.columns)
        target_names = np.array(['UHC','HC'])

        #dataset["CLASS_LABEL"]=dataset["CLASS_LABEL"].astype("category")
        #dataset["CLASS_LABEL"]=dataset["CLASS_LABEL"].cat.codes

        #print(dataset["CLASS_LABEL"])
        data = np.array(dataset.iloc[:, 0:104])
        target = np.array(dataset['CLASS_LABEL'])
        #print(big_df.activity.value_counts())
        
        self.data = Bunch(data=data, target=target,
                          target_names=target_names,feature_names=feature_names)

