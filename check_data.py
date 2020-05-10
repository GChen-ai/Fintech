import pandas as pd
import numpy as np

data=pd.read_csv('data/trd_and_tag_train_data.csv')
print(data['month_5'].value_counts())