import pandas as pd
import numpy as np

train_data=pd.read_csv('train_data/训练数据集_tag.csv')
test_data=pd.read_csv('评分数据集_tag.csv')
train_data.drop(columns='atdd_type',inplace=True)
train_data.drop(columns='deg_cd',inplace=True)
train_data.drop(columns='edu_deg_cd',inplace=True)

test_data.drop(columns='atdd_type',inplace=True)
test_data.drop(columns='deg_cd',inplace=True)
test_data.drop(columns='edu_deg_cd',inplace=True)
#test中atdd_type无\N，train中有7个去掉
#train_data=train_data[train_data['atdd_type']!='\\N']
train_data=train_data[train_data['hav_car_grp_ind']!='\\N']
test_data=test_data[test_data['hav_car_grp_ind']!='\\N']

#不存在信用卡借记卡数同时为0的数据
#print(test_data[(test_data['cur_credit_cnt']==0) & (test_data['cur_debit_cnt']==0)])

train_data['cur_debit_cnt'][train_data['cur_debit_cnt']>5]=5
test_data['cur_debit_cnt'][test_data['cur_debit_cnt']>5]=5

train_data['cur_credit_cnt'][train_data['cur_credit_cnt']>5]=5
test_data['cur_credit_cnt'][test_data['cur_credit_cnt']>5]=5
#把信用卡和借记卡张数大于5的值置为5
train_data['hav_car_grp_ind'][train_data['hav_car_grp_ind']==5]='1'
test_data['hav_car_grp_ind'][test_data['hav_car_grp_ind']==5]='1'
temp=train_data[train_data['gdr_cd']=='\\N']#学位为\N的人借记卡数基本为0
#找最相近的数据填补学历等信息
for i in range(temp.shape[0]):
    cur_credit_cnt=temp.iloc[i]['cur_credit_cnt']
    l1y_crd_card_csm_amt_dlm_cd=temp.iloc[i]['l1y_crd_card_csm_amt_dlm_cd']
    perm_crd_lmt_cd=temp.iloc[i]['perm_crd_lmt_cd']
    job_year=temp.iloc[i]['job_year']
    l12mon_buy_fin_mng_whl_tms=temp.iloc[i]['l12mon_buy_fin_mng_whl_tms']
    cust_inv_rsk_endu_lvl_cd=temp.iloc[i]['cust_inv_rsk_endu_lvl_cd']
    hav_car_grp_ind=temp.iloc[i]['hav_car_grp_ind']
    tot_ast_lvl_cd=temp.iloc[i]['tot_ast_lvl_cd']
    if len(train_data[(train_data['gdr_cd']!='\\N')& (train_data['cur_credit_cnt']==cur_credit_cnt)&(train_data['l1y_crd_card_csm_amt_dlm_cd']==l1y_crd_card_csm_amt_dlm_cd)&(train_data['perm_crd_lmt_cd']==perm_crd_lmt_cd)&(train_data['l12mon_buy_fin_mng_whl_tms']==l12mon_buy_fin_mng_whl_tms)&(train_data['cust_inv_rsk_endu_lvl_cd']==cust_inv_rsk_endu_lvl_cd)&(train_data['hav_car_grp_ind']==hav_car_grp_ind)&(train_data['tot_ast_lvl_cd']==tot_ast_lvl_cd)]['gdr_cd'].mode())!=0:

        temp.iloc[i]['gdr_cd']=train_data[(train_data['gdr_cd']!='\\N')& (train_data['cur_credit_cnt']==cur_credit_cnt)&(train_data['l1y_crd_card_csm_amt_dlm_cd']==l1y_crd_card_csm_amt_dlm_cd)&(train_data['perm_crd_lmt_cd']==perm_crd_lmt_cd)&(train_data['l12mon_buy_fin_mng_whl_tms']==l12mon_buy_fin_mng_whl_tms)&(train_data['cust_inv_rsk_endu_lvl_cd']==cust_inv_rsk_endu_lvl_cd)&(train_data['hav_car_grp_ind']==hav_car_grp_ind)&(train_data['tot_ast_lvl_cd']==tot_ast_lvl_cd)]['gdr_cd'].mode()[0]

        temp.iloc[i]['mrg_situ_cd']=train_data[(train_data['gdr_cd']!='\\N')& (train_data['cur_credit_cnt']==cur_credit_cnt)&(train_data['l1y_crd_card_csm_amt_dlm_cd']==l1y_crd_card_csm_amt_dlm_cd)&(train_data['perm_crd_lmt_cd']==perm_crd_lmt_cd)&(train_data['l12mon_buy_fin_mng_whl_tms']==l12mon_buy_fin_mng_whl_tms)&(train_data['cust_inv_rsk_endu_lvl_cd']==cust_inv_rsk_endu_lvl_cd)&(train_data['hav_car_grp_ind']==hav_car_grp_ind)&(train_data['tot_ast_lvl_cd']==tot_ast_lvl_cd)]['mrg_situ_cd'].mode()[0]


        temp.iloc[i]['acdm_deg_cd']=train_data[(train_data['gdr_cd']!='\\N')& (train_data['cur_credit_cnt']==cur_credit_cnt)&(train_data['l1y_crd_card_csm_amt_dlm_cd']==l1y_crd_card_csm_amt_dlm_cd)&(train_data['perm_crd_lmt_cd']==perm_crd_lmt_cd)&(train_data['l12mon_buy_fin_mng_whl_tms']==l12mon_buy_fin_mng_whl_tms)&(train_data['cust_inv_rsk_endu_lvl_cd']==cust_inv_rsk_endu_lvl_cd)&(train_data['hav_car_grp_ind']==hav_car_grp_ind)&(train_data['tot_ast_lvl_cd']==tot_ast_lvl_cd)]['acdm_deg_cd'].mode()[0]
    else:
        temp.iloc[i]['gdr_cd']=train_data[(train_data['gdr_cd']!='\\N')]['gdr_cd'].mode()[0]
        temp.iloc[i]['mrg_situ_cd']=train_data[(train_data['gdr_cd']!='\\N')]['mrg_situ_cd'].mode()[0]
        temp.iloc[i]['acdm_deg_cd']=train_data[(train_data['gdr_cd']!='\\N')]['acdm_deg_cd'].mode()[0]
train_data[train_data['gdr_cd']=='\\N']=temp
temp=test_data[test_data['gdr_cd']=='\\N']
for i in range(temp.shape[0]):
    cur_credit_cnt=temp.iloc[i]['cur_credit_cnt']
    l1y_crd_card_csm_amt_dlm_cd=temp.iloc[i]['l1y_crd_card_csm_amt_dlm_cd']
    perm_crd_lmt_cd=temp.iloc[i]['perm_crd_lmt_cd']
    job_year=temp.iloc[i]['job_year']
    l12mon_buy_fin_mng_whl_tms=temp.iloc[i]['l12mon_buy_fin_mng_whl_tms']
    cust_inv_rsk_endu_lvl_cd=temp.iloc[i]['cust_inv_rsk_endu_lvl_cd']
    hav_car_grp_ind=temp.iloc[i]['hav_car_grp_ind']
    tot_ast_lvl_cd=temp.iloc[i]['tot_ast_lvl_cd']
    if len(train_data[(train_data['gdr_cd']!='\\N')& (train_data['cur_credit_cnt']==cur_credit_cnt)&(train_data['l1y_crd_card_csm_amt_dlm_cd']==l1y_crd_card_csm_amt_dlm_cd)&(train_data['perm_crd_lmt_cd']==perm_crd_lmt_cd)&(train_data['l12mon_buy_fin_mng_whl_tms']==l12mon_buy_fin_mng_whl_tms)&(train_data['cust_inv_rsk_endu_lvl_cd']==cust_inv_rsk_endu_lvl_cd)&(train_data['hav_car_grp_ind']==hav_car_grp_ind)&(train_data['tot_ast_lvl_cd']==tot_ast_lvl_cd)]['gdr_cd'].mode())!=0:

        temp.iloc[i]['gdr_cd']=train_data[(train_data['gdr_cd']!='\\N')& (train_data['cur_credit_cnt']==cur_credit_cnt)&(train_data['l1y_crd_card_csm_amt_dlm_cd']==l1y_crd_card_csm_amt_dlm_cd)&(train_data['perm_crd_lmt_cd']==perm_crd_lmt_cd)&(train_data['l12mon_buy_fin_mng_whl_tms']==l12mon_buy_fin_mng_whl_tms)&(train_data['cust_inv_rsk_endu_lvl_cd']==cust_inv_rsk_endu_lvl_cd)&(train_data['hav_car_grp_ind']==hav_car_grp_ind)&(train_data['tot_ast_lvl_cd']==tot_ast_lvl_cd)]['gdr_cd'].mode()[0]

        temp.iloc[i]['mrg_situ_cd']=train_data[(train_data['gdr_cd']!='\\N')& (train_data['cur_credit_cnt']==cur_credit_cnt)&(train_data['l1y_crd_card_csm_amt_dlm_cd']==l1y_crd_card_csm_amt_dlm_cd)&(train_data['perm_crd_lmt_cd']==perm_crd_lmt_cd)&(train_data['l12mon_buy_fin_mng_whl_tms']==l12mon_buy_fin_mng_whl_tms)&(train_data['cust_inv_rsk_endu_lvl_cd']==cust_inv_rsk_endu_lvl_cd)&(train_data['hav_car_grp_ind']==hav_car_grp_ind)&(train_data['tot_ast_lvl_cd']==tot_ast_lvl_cd)]['mrg_situ_cd'].mode()[0]


        temp.iloc[i]['acdm_deg_cd']=train_data[(train_data['gdr_cd']!='\\N')& (train_data['cur_credit_cnt']==cur_credit_cnt)&(train_data['l1y_crd_card_csm_amt_dlm_cd']==l1y_crd_card_csm_amt_dlm_cd)&(train_data['perm_crd_lmt_cd']==perm_crd_lmt_cd)&(train_data['l12mon_buy_fin_mng_whl_tms']==l12mon_buy_fin_mng_whl_tms)&(train_data['cust_inv_rsk_endu_lvl_cd']==cust_inv_rsk_endu_lvl_cd)&(train_data['hav_car_grp_ind']==hav_car_grp_ind)&(train_data['tot_ast_lvl_cd']==tot_ast_lvl_cd)]['acdm_deg_cd'].mode()[0]
    else:
        temp.iloc[i]['gdr_cd']=train_data[(train_data['gdr_cd']!='\\N')]['gdr_cd'].mode()[0]
        temp.iloc[i]['mrg_situ_cd']=train_data[(train_data['gdr_cd']!='\\N')]['mrg_situ_cd'].mode()[0]
        temp.iloc[i]['acdm_deg_cd']=train_data[(train_data['gdr_cd']!='\\N')]['acdm_deg_cd'].mode()[0]
print(temp.shape)
print(test_data[test_data['gdr_cd']=='\\N'].shape)
print(test_data[test_data['gdr_cd']!='\\N'].shape)
test_data[test_data['gdr_cd']=='\\N']=temp
test_data['gdr_cd'][test_data['gdr_cd']=='\\N']=test_data['gdr_cd'].mode()[0]
test_data['mrg_situ_cd'][test_data['mrg_situ_cd']=='\\N']=test_data['mrg_situ_cd'].mode()[0]
test_data['acdm_deg_cd'][test_data['acdm_deg_cd']=='\\N']=test_data['acdm_deg_cd'].mode()[0]
train_data=train_data[train_data['acdm_deg_cd']!='\\N']
train_data.to_csv('data/clean_train_tag.csv',index=False)
test_data.to_csv('data/clean_test_tag.csv',index=False)