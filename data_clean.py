import pandas as pd
import numpy as np

train_data=pd.read_csv('train_data/训练数据集_tag.csv',index_col=0)
test_data=pd.read_csv('评分数据集_tag_b.csv',index_col=0)

train_data['atdd_type'].fillna(value=-1,inplace=True)
train_data['atdd_type'][train_data['atdd_type']=='\\N']=train_data['atdd_type'].mode()[0]
test_data['atdd_type'].fillna(value=-1,inplace=True)
test_data['atdd_type'][test_data['atdd_type']=='\\N']=test_data['atdd_type'].mode()[0]

train_data['deg_cd'].fillna(value=-1,inplace=True)
train_data['deg_cd'][train_data['deg_cd']=='\\N']=train_data['deg_cd'].mode()[0]
test_data['deg_cd'].fillna(value=-1,inplace=True)
test_data['deg_cd'][test_data['deg_cd']=='\\N']=test_data['deg_cd'].mode()[0]

train_data['edu_deg_cd'].fillna(value=-1,inplace=True)
train_data['edu_deg_cd'][train_data['edu_deg_cd']=='\\N']=train_data['edu_deg_cd'].mode()[0]
test_data['edu_deg_cd'].fillna(value=-1,inplace=True)
test_data['edu_deg_cd'][test_data['edu_deg_cd']=='\\N']=test_data['edu_deg_cd'].mode()[0]
#train_data.drop(columns='deg_cd',inplace=True)
#train_data.drop(columns='edu_deg_cd',inplace=True)

#test_data.drop(columns='deg_cd',inplace=True)
#test_data.drop(columns='edu_deg_cd',inplace=True)
#test中atdd_type无\N，train中有7个去掉
#train_data=train_data[train_data['atdd_type']!='\\N']
train_data=train_data[train_data['hav_car_grp_ind']!='\\N']
test_data=test_data[test_data['hav_car_grp_ind']!='\\N']

#不存在信用卡借记卡数同时为0的数据
#print(test_data[(test_data['cur_credit_cnt']==0) & (test_data['cur_debit_cnt']==0)])
train_data['cur_cnt']=0
test_data['cur_cnt']=0
train_data['cur_debit_cnt'][train_data['cur_debit_cnt']>50]=50
test_data['cur_debit_cnt'][test_data['cur_debit_cnt']>50]=50

train_data['cur_credit_cnt'][train_data['cur_credit_cnt']>50]=50
test_data['cur_credit_cnt'][test_data['cur_credit_cnt']>50]=50

train_data['cur_cnt']=train_data['cur_debit_cnt']+train_data['cur_credit_cnt']
test_data['cur_cnt']=test_data['cur_debit_cnt']+test_data['cur_credit_cnt']
#把信用卡和借记卡张数大于5的值置为5
train_data['hav_car_grp_ind'][train_data['hav_car_grp_ind']==5]='1'
test_data['hav_car_grp_ind'][test_data['hav_car_grp_ind']==5]='1'
train_data['hav_car_grp_ind']=train_data['hav_car_grp_ind'].astype('category')
test_data['hav_car_grp_ind']=test_data['hav_car_grp_ind'].astype('category')
temp=train_data[train_data['gdr_cd']=='\\N']#学位为\N的人借记卡数基本为0
#找最相近的数据填补学历等信息
count1=0
count2=0
for index in temp.index:
    cur_credit_cnt=temp['cur_credit_cnt'].loc[index]
    l1y_crd_card_csm_amt_dlm_cd=temp['l1y_crd_card_csm_amt_dlm_cd'].loc[index]
    perm_crd_lmt_cd=temp['perm_crd_lmt_cd'].loc[index]
    job_year=temp['job_year'].loc[index]
    l12mon_buy_fin_mng_whl_tms=temp['l12mon_buy_fin_mng_whl_tms'].loc[index]
    cust_inv_rsk_endu_lvl_cd=temp['cust_inv_rsk_endu_lvl_cd'].loc[index]
    hav_car_grp_ind=temp['hav_car_grp_ind'].loc[index]
    tot_ast_lvl_cd=temp['tot_ast_lvl_cd'].loc[index]
    cur_train_sample=train_data[(train_data['gdr_cd']!='\\N')& (train_data['cur_credit_cnt']==cur_credit_cnt)&(train_data['l1y_crd_card_csm_amt_dlm_cd']==l1y_crd_card_csm_amt_dlm_cd)&(train_data['perm_crd_lmt_cd']==perm_crd_lmt_cd)&(train_data['l12mon_buy_fin_mng_whl_tms']==l12mon_buy_fin_mng_whl_tms)&(train_data['cust_inv_rsk_endu_lvl_cd']==cust_inv_rsk_endu_lvl_cd)&(train_data['hav_car_grp_ind']==hav_car_grp_ind)&(train_data['tot_ast_lvl_cd']==tot_ast_lvl_cd)]
    if cur_train_sample.shape[0]!=0:
        temp['gdr_cd'].loc[index]=cur_train_sample['gdr_cd'].mode()[0]
        temp['mrg_situ_cd'].loc[index]=cur_train_sample['mrg_situ_cd'].mode()[0]
        temp['acdm_deg_cd'].loc[index]=cur_train_sample['acdm_deg_cd'].mode()[0]
    else:
        count1+=1
        temp['gdr_cd'].loc[index]=train_data[(train_data['gdr_cd']!='\\N')]['gdr_cd'].mode()[0]
        temp['mrg_situ_cd'].loc[index]=train_data[(train_data['gdr_cd']!='\\N')]['mrg_situ_cd'].mode()[0]
        temp['acdm_deg_cd'].loc[index]=train_data[(train_data['gdr_cd']!='\\N')]['acdm_deg_cd'].mode()[0]
train_data[train_data['gdr_cd']=='\\N']=temp
temp=test_data[test_data['gdr_cd']=='\\N']
for index in temp.index:
    cur_credit_cnt=temp['cur_credit_cnt'].loc[index]
    l1y_crd_card_csm_amt_dlm_cd=temp['l1y_crd_card_csm_amt_dlm_cd'].loc[index]
    perm_crd_lmt_cd=temp['perm_crd_lmt_cd'].loc[index]
    job_year=temp['job_year'].loc[index]
    l12mon_buy_fin_mng_whl_tms=temp['l12mon_buy_fin_mng_whl_tms'].loc[index]
    cust_inv_rsk_endu_lvl_cd=temp['cust_inv_rsk_endu_lvl_cd'].loc[index]
    hav_car_grp_ind=temp['hav_car_grp_ind'].loc[index]
    tot_ast_lvl_cd=temp['tot_ast_lvl_cd'].loc[index]
    cur_train_sample=train_data[(train_data['gdr_cd']!='\\N')& (train_data['cur_credit_cnt']==cur_credit_cnt)&(train_data['l1y_crd_card_csm_amt_dlm_cd']==l1y_crd_card_csm_amt_dlm_cd)&(train_data['perm_crd_lmt_cd']==perm_crd_lmt_cd)&(train_data['l12mon_buy_fin_mng_whl_tms']==l12mon_buy_fin_mng_whl_tms)&(train_data['cust_inv_rsk_endu_lvl_cd']==cust_inv_rsk_endu_lvl_cd)&(train_data['hav_car_grp_ind']==hav_car_grp_ind)&(train_data['tot_ast_lvl_cd']==tot_ast_lvl_cd)]

    cur_test_sample=test_data[(test_data['gdr_cd']!='\\N')& (test_data['cur_credit_cnt']==cur_credit_cnt)&(test_data['l1y_crd_card_csm_amt_dlm_cd']==l1y_crd_card_csm_amt_dlm_cd)&(test_data['perm_crd_lmt_cd']==perm_crd_lmt_cd)&(test_data['l12mon_buy_fin_mng_whl_tms']==l12mon_buy_fin_mng_whl_tms)&(test_data['cust_inv_rsk_endu_lvl_cd']==cust_inv_rsk_endu_lvl_cd)&(test_data['hav_car_grp_ind']==hav_car_grp_ind)&(train_data['tot_ast_lvl_cd']==tot_ast_lvl_cd)]
    if cur_test_sample.shape[0]!=0:
        temp['gdr_cd'].loc[index]=cur_test_sample['gdr_cd'].mode()[0]
        temp['mrg_situ_cd'].loc[index]=cur_test_sample['mrg_situ_cd'].mode()[0]
        temp['acdm_deg_cd'].loc[index]=cur_test_sample['acdm_deg_cd'].mode()[0]
    elif cur_train_sample.shape[0]!=0:
        temp['gdr_cd'].loc[index]=cur_train_sample['gdr_cd'].mode()[0]
        temp['mrg_situ_cd'].loc[index]=cur_train_sample['mrg_situ_cd'].mode()[0]
        temp['acdm_deg_cd'].loc[index]=cur_train_sample['acdm_deg_cd'].mode()[0]
    else:
        count2+=1
        temp['gdr_cd'].loc[index]=train_data[(train_data['gdr_cd']!='\\N')]['gdr_cd'].mode()[0]
        temp['mrg_situ_cd'].loc[index]=train_data[(train_data['gdr_cd']!='\\N')]['mrg_situ_cd'].mode()[0]
        temp['acdm_deg_cd'].loc[index]=train_data[(train_data['gdr_cd']!='\\N')]['acdm_deg_cd'].mode()[0]
print(count1)
print(count2)
test_data[test_data['gdr_cd']=='\\N']=temp
test_data['gdr_cd'][test_data['gdr_cd']=='\\N']=test_data['gdr_cd'].mode()[0]
test_data['mrg_situ_cd'][test_data['mrg_situ_cd']=='\\N']=test_data['mrg_situ_cd'].mode()[0]
test_data['acdm_deg_cd'][test_data['acdm_deg_cd']=='\\N']=test_data['acdm_deg_cd'].mode()[0]
train_data=train_data[train_data['acdm_deg_cd']!='\\N']
train_data.to_csv('data/clean_train_tag.csv')
test_data.to_csv('data/clean_test_tag.csv')