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
train_data=pd.read_csv('data/train_tag.csv',index_col=0)
test_data=pd.read_csv('data/test_tag.csv',index_col=0)

#train_1=train_data[train_data['flag']==1]
#train_0=train_data[train_data['flag']==0].sample(n=10000)
#train_data=pd.concat([train_1,train_0])


label=train_data['flag']
train_data.drop(columns='flag',inplace=True)
train_data,test_data=train_data.align(test_data,join='left',axis=1)

train_X,test_X, train_y, test_y = train_test_split(train_data,
                                                   label,
                                                   test_size = 0.2)
clf=XGBClassifier(max_depth=4, learning_rate=0.1, n_estimators=500, silent=True, objective='binary:logistic')
clf.fit(train_X,train_y)
predict_y1=clf.predict_proba(test_X)[:,1]
result1=sklearn.metrics.roc_auc_score(test_y,predict_y1)
Y1=clf.predict_proba(test_data)[:,1]


clf2=lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',metrics='auc',learning_rate=0.01, n_estimators=1000, max_depth=4, num_leaves=10,max_bin=15,min_data_in_leaf=51,bagging_fraction=0.6,bagging_freq= 0, feature_fraction= 0.8,lambda_l1=1e-05,lambda_l2=1e-05,min_split_gain=0)
clf2.fit(train_X,train_y)
predict_y2=clf2.predict_proba(test_X)[:,1]
result2=sklearn.metrics.roc_auc_score(test_y,predict_y2)
print(result1)
print(result2)
predict_y=(predict_y1+predict_y2)/2
result=sklearn.metrics.roc_auc_score(test_y,predict_y)
print(result)
Y2=clf2.predict_proba(test_data)[:,1]
Y=(Y1+Y2)/2
test_data['flag']=Y
test_data=test_data['flag']
test_data.to_csv('result.txt', sep='\t',header=False)
#plot_importance(clf)
#pyplot.show()

