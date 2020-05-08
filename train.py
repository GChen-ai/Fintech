import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import math
train_data=pd.read_csv('train_tag.csv',index_col=0)
test_data=pd.read_csv('test_tag.csv',index_col=0)
label=train_data['flag']
train_data.drop(columns='flag',inplace=True)
train_data,test_data=train_data.align(test_data,join='left',axis=1)

train_X,test_X, train_y, test_y = train_test_split(train_data,
                                                   label,
                                                   test_size = 0.2,
                                                   random_state = 0)
n_features=train_data.shape[1]
clf=RandomForestClassifier(n_estimators=100,n_jobs=-1,max_features='sqrt')
clf.fit(train_X,train_y)
predict_y=clf.predict_proba(test_X)[:,1]
result=sklearn.metrics.roc_auc_score(test_y,predict_y)
Y=clf.predict_proba(test_data.values)
print(Y)