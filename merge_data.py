import pandas as pd
import numpy as np

trd_train_data=pd.read_csv('data/train_trd.csv')
trd_test_data=pd.read_csv('data/test_trd.csv')

tag_train_data=pd.read_csv('data/train_tag.csv')
tag_test_data=pd.read_csv('data/test_tag.csv')

train_data=pd.merge(tag_train_data,trd_train_data,on='id')
test_data=pd.merge(tag_test_data,trd_test_data,on='id')
tag_test_data=tag_test_data[~tag_test_data['id'].isin(test_data['id'])]

train_data.to_csv('data/trd_and_tag_train_data.csv',index=False)
test_data.to_csv('data/trd_and_tag_test_data.csv',index=False)

tag_test_data.to_csv('data/test_tag.csv',index=False)