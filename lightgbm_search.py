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
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
train_data=pd.read_csv('data/all_train_data.csv',index_col=0)
test_data=pd.read_csv('data/all_test_data.csv',index_col=0)
#train_1=train_data[train_data['flag']==1]
#train_0=train_data[train_data['flag']==0].sample(n=20000)
#train_data=pd.concat([train_1,train_0])


label=train_data['flag']
train_data.drop(columns='flag',inplace=True)
train_data,test_data=train_data.align(test_data,join='left',axis=1)
#train_X,test_X, train_y, test_y = train_test_split(train_data,label,test_size = 0.2)
parameters ={'max_bin': range(5,256,10), 'min_data_in_leaf':range(1,102,10)}





gbm=lgb.LGBMClassifier(objective = 'binary',is_unbalance = True,metric = 'auc',max_depth = 4,lambda_l2=0.3,lambda_l1=0.9,num_leaves = 15,learning_rate = 0.1,feature_fraction = 1.0,min_child_samples=5,min_child_weight=0.001,bagging_fraction = 0.7,bagging_freq = 60,cat_smooth = 0,num_iterations = 500,max_bin=5, min_data_in_leaf=61,min_split_gain=0)#RandomForestClassifier(n_estimators=500,n_jobs=-1,max_features='sqrt')
gsearch = GridSearchCV(gbm, param_grid=parameters, scoring='roc_auc', cv=5)
gsearch.fit(train_data, label)
print('参数的最佳取值:{0}'.format(gsearch.best_params_))
print('最佳模型得分:{0}'.format(gsearch.best_score_))
print(gsearch.cv_results_['mean_test_score'])
print(gsearch.cv_results_['params'])
