import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import math
import random
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.model_selection import KFold
from catboost import CatBoostClassifier
import lightgbm as lgb
train_data=pd.read_csv('data/trd_and_tag_train_data.csv',index_col=0)
test_data=pd.read_csv('data/trd_and_tag_test_data.csv')
test_data=test_data.set_index('id')
#train_1=train_data[train_data['flag']==1]
#train_0=train_data[train_data['flag']==0].sample(n=20000)
#train_data=pd.concat([train_1,train_0])


label=train_data['flag']
train_data.drop(columns='flag',inplace=True)
train_data,test_data=train_data.align(test_data,join='left',axis=1)
train_X,test_X, train_y, test_y = train_test_split(train_data,
                                                   label,
                                                   test_size = 0.1)
clf=lgb.LGBMClassifier(objective = 'binary',metric = 'auc',max_depth = 3,lambda_l2=0.3,lambda_l1=0.7,num_leaves = 5,learning_rate = 0.05,feature_fraction = 0.9,min_child_samples=5,min_child_weight=0.001,bagging_fraction = 0.6,bagging_freq = 35,cat_smooth = 0,num_iterations = 5000,max_bin=15, min_data_in_leaf=101,min_split_gain=0)#CatBoostClassifier(learning_rate=0.1,depth=3,objective = 'Logloss',custom_metric = 'AUC',n_estimators = 1000,reg_lambda=0.3)#
clf.fit(train_X,train_y)
predict_y1=clf.predict_proba(test_X)[:,1]
result1=sklearn.metrics.roc_auc_score(test_y,predict_y1)
print(result1)
Y1=clf.predict_proba(test_data)[:,1]

test_data['flag']=Y1
test_data=test_data['flag']
test_data.to_csv('result_trd.txt', sep='\t',header=False)
#plot_importance(clf)
#pyplot.show()