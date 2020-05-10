import pandas as pd
import numpy as np

trd_train_data=pd.read_csv('data/train_trd.csv')
trd_test_data=pd.read_csv('data/test_trd.csv')

tag_train_data=pd.read_csv('data/train_tag.csv')
tag_test_data=pd.read_csv('data/test_tag.csv')

train_data=pd.merge(tag_train_data,trd_train_data,on='id')
test_data=pd.merge(tag_test_data,trd_test_data,on='id')

train_data['B_amt_min']=(train_data['B_amt_min']-train_data['B_amt_min'].min())/(train_data['B_amt_min'].max()-train_data['B_amt_min'].min())
train_data['B_amt_max']=(train_data['B_amt_max']-train_data['B_amt_max'].min())/(train_data['B_amt_max'].max()-train_data['B_amt_max'].min())
train_data['B_amt_mean']=(train_data['B_amt_mean']-train_data['B_amt_mean'].min())/(train_data['B_amt_mean'].max()-train_data['B_amt_mean'].min())
train_data['B_amt_Range']=(train_data['B_amt_Range']-train_data['B_amt_Range'].min())/(train_data['B_amt_Range'].max()-train_data['B_amt_Range'].min())
train_data['B_amt_count']=(train_data['B_amt_count']-train_data['B_amt_count'].min())/(train_data['B_amt_count'].max()-train_data['B_amt_count'].min())

train_data['C_amt_min']=(train_data['C_amt_min']-train_data['C_amt_min'].min())/(train_data['C_amt_min'].max()-train_data['C_amt_min'].min())
train_data['C_amt_max']=(train_data['C_amt_max']-train_data['C_amt_max'].min())/(train_data['C_amt_max'].max()-train_data['C_amt_max'].min())
train_data['C_amt_mean']=(train_data['C_amt_mean']-train_data['C_amt_mean'].min())/(train_data['C_amt_mean'].max()-train_data['C_amt_mean'].min())
train_data['C_amt_Range']=(train_data['C_amt_Range']-train_data['C_amt_Range'].min())/(train_data['C_amt_Range'].max()-train_data['C_amt_Range'].min())
train_data['C_amt_count']=(train_data['C_amt_count']-train_data['C_amt_count'].min())/(train_data['C_amt_count'].max()-train_data['C_amt_count'].min())

test_data['B_amt_min']=(test_data['B_amt_min']-test_data['B_amt_min'].min())/(test_data['B_amt_min'].max()-test_data['B_amt_min'].min())
test_data['B_amt_max']=(test_data['B_amt_max']-test_data['B_amt_max'].min())/(test_data['B_amt_max'].max()-test_data['B_amt_max'].min())
test_data['B_amt_mean']=(test_data['B_amt_mean']-test_data['B_amt_mean'].min())/(test_data['B_amt_mean'].max()-test_data['B_amt_mean'].min())
test_data['B_amt_Range']=(test_data['B_amt_Range']-test_data['B_amt_Range'].min())/(test_data['B_amt_Range'].max()-test_data['B_amt_Range'].min())
test_data['B_amt_count']=(test_data['B_amt_count']-test_data['B_amt_count'].min())/(test_data['B_amt_count'].max()-test_data['B_amt_count'].min())

test_data['C_amt_min']=(test_data['C_amt_min']-test_data['C_amt_min'].min())/(test_data['C_amt_min'].max()-test_data['C_amt_min'].min())
test_data['C_amt_max']=(test_data['C_amt_max']-test_data['C_amt_max'].min())/(test_data['C_amt_max'].max()-test_data['C_amt_max'].min())
test_data['C_amt_mean']=(test_data['C_amt_mean']-test_data['C_amt_mean'].min())/(test_data['C_amt_mean'].max()-test_data['C_amt_mean'].min())
test_data['C_amt_Range']=(test_data['C_amt_Range']-test_data['C_amt_Range'].min())/(test_data['C_amt_Range'].max()-test_data['C_amt_Range'].min())
test_data['C_amt_count']=(test_data['C_amt_count']-train_data['C_amt_count'].min())/(test_data['C_amt_count'].max()-test_data['C_amt_count'].min())

tag_test_data=tag_test_data[~tag_test_data['id'].isin(test_data['id'])]


train_data.to_csv('data/trd_and_tag_train_data.csv',index=False)
test_data.to_csv('data/trd_and_tag_test_data.csv',index=False)

tag_test_data.to_csv('data/test_tag.csv',index=False)