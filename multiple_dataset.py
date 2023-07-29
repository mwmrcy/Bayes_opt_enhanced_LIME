import glob
import pandas as pd
import numpy as np
from sklearn.utils import Bunch

# Get CSV files list from a folder
class LoadDataset:
    def __init__(self):
        path = "data/archive"
        csv_files = glob.glob(path + "/*.csv")

        # Read each CSV file into DataFrame
        # This creates a list of dataframes
        df_list = (pd.read_csv(file) for file in csv_files)

        # Concatenate all DataFrames
        big_df   = pd.concat(df_list, ignore_index=True)

        #print(big_df.loc[:,"activity"])
        #print(big_df.activity.value_counts())
        big_df = big_df.iloc[:,1:]
        feature_names = list(big_df.columns)
        target_names = np.array(['down_by_elevator','going_down', 'going_up','running','sitting','sitting_down','standing','standing_up','up_by_elevator','walking'])

        big_df["activity"]=big_df["activity"].astype("category")
        big_df["activity"]=big_df["activity"].cat.codes

        #print(big_df["activity"])

        data = np.array(big_df.iloc[:, 0:37])
        target = np.array(big_df['activity'])

        #print(big_df.activity.value_counts())

        self.data = Bunch(data=data, target=target,target_names=target_names,feature_names=feature_names)

