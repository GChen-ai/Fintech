import pandas as pd
import numpy as np

trd_train_data=pd.read_csv('data/train_trd.csv')
trd_test_data=pd.read_csv('data/test_trd.csv')

tag_train_data=pd.read_csv('data/train_tag.csv')
tag_test_data=pd.read_csv('data/test_tag.csv')


train_data=pd.merge(tag_train_data,trd_train_data,on='id')
test_data=pd.merge(tag_test_data,trd_test_data,on='id')
rest_train=tag_train_data[~tag_train_data['id'].isin(train_data['id'])]
rest_test=tag_test_data[~tag_test_data['id'].isin(test_data['id'])]
rest_train=rest_train.set_index('id')
rest_test=rest_test.set_index('id')
tag_train_data.drop(columns='id',inplace=True)
tag_test_data.drop(columns='id',inplace=True)
for col in trd_train_data.columns:
    rest_train[col]=0
    rest_test[col]=0
count1=0
count2=0
'''for index in rest_train.index:
    cur_credit_cnt=rest_train['cur_credit_cnt'].loc[index]
    cur_debit_cnt=rest_train['cur_debit_cnt'].loc[index]
    cust_inv_rsk_endu_lvl_cd=rest_train['cust_inv_rsk_endu_lvl_cd'].loc[index]
    hav_car_grp_ind=rest_train['hav_car_grp_ind'].loc[index]
    cur_debit_min_opn_dt_cnt_new=rest_train['cur_debit_min_opn_dt_cnt_new'].loc[index]
    cur_credit_min_opn_dt_cnt_new=rest_train['cur_credit_min_opn_dt_cnt_new'].loc[index]
    cur_train_sample=train_data[(train_data['cur_debit_cnt']==cur_debit_cnt)&(train_data['cur_credit_cnt']==cur_credit_cnt)&(train_data['cust_inv_rsk_endu_lvl_cd']==cust_inv_rsk_endu_lvl_cd)&(train_data['hav_car_grp_ind']==hav_car_grp_ind)&(train_data['cur_debit_min_opn_dt_cnt_new']==cur_debit_min_opn_dt_cnt_new)&(train_data['cur_credit_min_opn_dt_cnt_new']==cur_credit_min_opn_dt_cnt_new)]
    if cur_train_sample.shape[0]!=0:
        for col in trd_train_data.columns:
            if col!='id':
                rest_train[col].loc[index]=cur_train_sample[col].mean()
    else:
        count1+=1
        rest_train=rest_train.drop(index=index)'''

'''for index in rest_test.index:
    cur_credit_cnt=rest_test['cur_credit_cnt'].loc[index]
    cur_debit_cnt=rest_test['cur_debit_cnt'].loc[index]
    cust_inv_rsk_endu_lvl_cd=rest_test['cust_inv_rsk_endu_lvl_cd'].loc[index]
    hav_car_grp_ind=rest_test['hav_car_grp_ind'].loc[index]
    cur_debit_min_opn_dt_cnt_new=rest_test['cur_debit_min_opn_dt_cnt_new'].loc[index]
    cur_credit_min_opn_dt_cnt_new=rest_test['cur_credit_min_opn_dt_cnt_new'].loc[index]
    cur_train_sample=train_data[(train_data['cur_debit_cnt']==cur_debit_cnt)&(train_data['cur_credit_cnt']==cur_credit_cnt)&(train_data['cust_inv_rsk_endu_lvl_cd']==cust_inv_rsk_endu_lvl_cd)&(train_data['hav_car_grp_ind']==hav_car_grp_ind)&(train_data['cur_debit_min_opn_dt_cnt_new']==cur_debit_min_opn_dt_cnt_new)&(train_data['cur_credit_min_opn_dt_cnt_new']==cur_credit_min_opn_dt_cnt_new)]
    cur_test_sample=test_data[(test_data['cur_debit_cnt']==cur_debit_cnt)&(test_data['cur_credit_cnt']==cur_credit_cnt)&(test_data['cust_inv_rsk_endu_lvl_cd']==cust_inv_rsk_endu_lvl_cd)&(test_data['hav_car_grp_ind']==hav_car_grp_ind)&(test_data['cur_debit_min_opn_dt_cnt_new']==cur_debit_min_opn_dt_cnt_new)&(test_data['cur_credit_min_opn_dt_cnt_new']==cur_credit_min_opn_dt_cnt_new)]
    if cur_test_sample.shape[0]!=0:
        for col in trd_train_data.columns:
            if col!='id':
                rest_test[col].loc[index]=cur_test_sample[col].mean()
    elif cur_train_sample.shape[0]!=0:
        for col in trd_train_data.columns:
            if col!='id':
                rest_test[col].loc[index]=cur_train_sample[col].mean()
    else:
        
        temp2=test_data[test_data['ovd_30d_loan_tot_cnt']==rest_test['ovd_30d_loan_tot_cnt'].loc[index]]
        if temp2.shape[0]!=0:
            for col in trd_train_data.columns:
                if col!='id':
                    rest_test[col].loc[index]=temp2[col].mean()
        else:
            count2+=1
            rest_test[col].loc[index]=cur_test_sample[col].mean()'''
train_data=train_data.set_index('id')
test_data=test_data.set_index('id')
#train_data,rest_train=train_data.align(rest_train,join='left',axis=1)
#test_data,rest_test=test_data.align(rest_test,join='left',axis=1)
#train_data=pd.concat([train_data,rest_train])
#test_data=pd.concat([test_data,rest_test])
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

print(count1)
print(count2)
print(train_data.shape)
train_data.to_csv('data/trd_and_tag_train_data.csv')
test_data.to_csv('data/trd_and_tag_test_data.csv')

