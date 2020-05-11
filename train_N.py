import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import math
import random
from xgboost import plot_importance
from matplotlib import pyplot
import lightgbm as lgb
from catboost import CatBoostClassifier
train_data=pd.read_csv('data/train_tag.csv',index_col=0)
test_data=pd.read_csv('评分数据集_tag_b.csv',index_col=0)
test_data=test_data[test_data['hav_car_grp_ind']=='\\N']

cols=[x for i,x in enumerate(test_data.columns) if test_data.iat[0,i]=='\\N']
test_data=test_data.drop(cols,axis=1)
test_data=test_data.drop('mrg_situ_cd',axis=1)
test_data=test_data.drop('acdm_deg_cd',axis=1)
test_data=test_data.drop(columns='atdd_type',axis=1)
test_data=test_data.drop(columns='deg_cd',axis=1)
test_data=test_data.drop(columns='edu_deg_cd',axis=1)

label=train_data['flag']
train_data=train_data[test_data.columns]
test_data['gdr_cd'][test_data['gdr_cd']=='F']=0
test_data['gdr_cd'][test_data['gdr_cd']=='M']=1
test_data['gdr_cd'][test_data['gdr_cd']=='\\N']=1
test_data['gdr_cd']=test_data['gdr_cd'].astype('float')
train_data,test_data=train_data.align(test_data,join='left',axis=1)

train_X,test_X, train_y, test_y = train_test_split(train_data,
                                                   label,
                                                   test_size = 0.1)
clf=CatBoostClassifier(learning_rate=0.1,depth=3,objective = 'Logloss',custom_metric = 'AUC',n_estimators = 1000,reg_lambda=0.3)#lgb.LGBMClassifier(objective = 'binary',metric = 'auc',max_depth = 3,lambda_l2=0.3,lambda_l1=0.7,num_leaves = 5,learning_rate = 0.05,feature_fraction = 0.9,min_child_samples=5,min_child_weight=0.001,bagging_fraction = 0.6,bagging_freq = 35,cat_smooth = 0,num_iterations = 1000,max_bin=15, min_data_in_leaf=101,min_split_gain=0)
clf.fit(train_X,train_y)
predict_y1=clf.predict_proba(test_X)[:,1]
result1=sklearn.metrics.roc_auc_score(test_y,predict_y1)
Y1=clf.predict_proba(test_data)[:,1]
print(result1)
test_data['flag']=Y1
test_data=test_data['flag']
test_data.to_csv('result_N.txt', sep='\t',header=False)
#plot_importance(clf)
#pyplot.show()

