import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import math
import random
train_data=pd.read_csv('data/trd_and_tag_train_data.csv',index_col=0)
test_data=pd.read_csv('data/trd_and_tag_test_data.csv',index_col=0)

train_1=train_data[train_data['flag']==1]
train_0=train_data[train_data['flag']==0].sample(n=10000)
train_data=pd.concat([train_1,train_0])


label=train_data['flag']
train_data.drop(columns='flag',inplace=True)
train_data,test_data=train_data.align(test_data,join='left',axis=1)

train_X,test_X, train_y, test_y = train_test_split(train_data,
                                                   label,
                                                   test_size = 0.2)
#clf=RandomForestClassifier(n_estimators=1000,n_jobs=-1,max_features='sqrt')
clf=XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=1000, 
                   silent=True, objective='binary:logistic')
clf.fit(train_X,train_y)
predict_y=clf.predict_proba(test_X)[:,1]
result=sklearn.metrics.roc_auc_score(test_y,predict_y)
print(result)
Y=clf.predict_proba(test_data)
test_data['flag']=Y[:,1]
print(Y)

test_data=test_data['flag']
test_data.to_csv('result_trd.txt', sep='\t',header=False)