import pandas as pd
import os
import numpy as np


df = pd.DataFrame()
dir_path = "Translator/data/"

for file in os.listdir(dir_path):
    x = pd.read_excel(dir_path + file, index_col=0)
    x = x.loc[:,["원문", "번역문"]]
    print(x)
    df = pd.concat([df, x], axis=0, sort=False)


print(df)
np.save(dir_path + "data.npy", df.to_numpy())
