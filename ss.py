import pandas as pd
import numpy as np


dataset= pd.read_csv('data/archive/GRF_data_to_use.csv')
        
dataset = dataset.iloc[:,1:]
feature_names = list(dataset.columns)
        #target_names = np.array(['down_by_elevator','going_down', 'going_up','running','sitting','sitting_down','standing','standing_up','up_by_elevator','walking'])
target_names = np.array(['A','C','H','HC','K'])

dataset["CLASS_LABEL"]=dataset["CLASS_LABEL"].astype("category")
dataset["CLASS_LABEL"]=dataset["CLASS_LABEL"].cat.codes

print(dataset["CLASS_LABEL"])
data = np.array(dataset.iloc[:, 0:113])
target = np.array(dataset['CLASS_LABEL'])
print(dataset.CLASS_LABEL.value_counts())
        
