import pandas as pd
import numpy as np

train_data=pd.read_csv('new_train_tag.csv')
test_data=pd.read_csv('new_test_tag.csv')
#性别转化为0 1
train_data['gdr_cd'][train_data['gdr_cd']=='F']=0
train_data['gdr_cd'][train_data['gdr_cd']=='M']=1

test_data['gdr_cd'][test_data['gdr_cd']=='F']=0
test_data['gdr_cd'][test_data['gdr_cd']=='M']=1
#'mrg_situ_cd'有6类，onehot
temp=pd.get_dummies(train_data['mrg_situ_cd'],prefix="mrg_situ_cd").astype('float')
train_data[temp.columns]=temp
train_data.drop(columns='mrg_situ_cd',inplace=True)

temp=pd.get_dummies(test_data['mrg_situ_cd'],prefix="mrg_situ_cd").astype('float')
test_data[temp.columns]=temp
test_data.drop(columns='mrg_situ_cd',inplace=True)
#acdm_deg_cd有8类,onehot
temp=pd.get_dummies(train_data['acdm_deg_cd'],prefix="acdm_deg_cd").astype('float')
train_data[temp.columns]=temp
train_data.drop(columns='acdm_deg_cd',inplace=True)

temp=pd.get_dummies(test_data['acdm_deg_cd'],prefix="acdm_deg_cd").astype('float')
test_data[temp.columns]=temp
test_data.drop(columns='acdm_deg_cd',inplace=True)

#新建特征是否为工资卡，是为1，否为0
train_data['frs_agn_dt_cnt']=train_data['frs_agn_dt_cnt'].astype('int')
train_data['frs_agn_dt_cls']=0
train_data['frs_agn_dt_cls'][train_data['frs_agn_dt_cnt']!=-1]=1
train_data['frs_agn_dt_cls'][train_data['frs_agn_dt_cnt']==-1]=0

test_data['frs_agn_dt_cnt']=test_data['frs_agn_dt_cnt'].astype('int')
test_data['frs_agn_dt_cls']=0
test_data['frs_agn_dt_cls'][test_data['frs_agn_dt_cnt']!=-1]=1
test_data['frs_agn_dt_cls'][test_data['frs_agn_dt_cnt']==-1]=0

#新建资产等级分类特征
train_data['tot_ast_lvl_cd']=train_data['tot_ast_lvl_cd'].astype('int')
train_data['tot_ast_lvl_cls']=0
train_data['tot_ast_lvl_cls'][train_data['tot_ast_lvl_cd']!=0]=1
train_data['tot_ast_lvl_cls'][train_data['tot_ast_lvl_cd']==0]=0

test_data['tot_ast_lvl_cd']=test_data['tot_ast_lvl_cd'].astype('int')
test_data['tot_ast_lvl_cls']=0
test_data['tot_ast_lvl_cls'][test_data['tot_ast_lvl_cd']!=0]=1
test_data['tot_ast_lvl_cls'][test_data['tot_ast_lvl_cd']==0]=0


#新建本年月均代发金额分层分类特征
train_data['bk1_cur_year_mon_avg_agn_amt_cd']=train_data['bk1_cur_year_mon_avg_agn_amt_cd'].astype('int')
train_data['bk1_cur_year_mon_avg_agn_amt_cd_cls']=0
train_data['bk1_cur_year_mon_avg_agn_amt_cd_cls'][train_data['bk1_cur_year_mon_avg_agn_amt_cd']!=0]=1
train_data['bk1_cur_year_mon_avg_agn_amt_cd_cls'][train_data['bk1_cur_year_mon_avg_agn_amt_cd']==0]=0

test_data['bk1_cur_year_mon_avg_agn_amt_cd']=test_data['bk1_cur_year_mon_avg_agn_amt_cd'].astype('int')
test_data['bk1_cur_year_mon_avg_agn_amt_cd_cls']=0
test_data['bk1_cur_year_mon_avg_agn_amt_cd_cls'][test_data['bk1_cur_year_mon_avg_agn_amt_cd']!=0]=1
test_data['bk1_cur_year_mon_avg_agn_amt_cd_cls'][test_data['bk1_cur_year_mon_avg_agn_amt_cd']==0]=0

#是否贷款逾期分类特征
train_data['his_lng_ovd_day']=train_data['his_lng_ovd_day'].astype('int')
train_data['his_lng_ovd_day_cls']=0
train_data['his_lng_ovd_day_cls'][train_data['his_lng_ovd_day']>0]=1
train_data['his_lng_ovd_day_cls'][train_data['his_lng_ovd_day']<=0]=0

test_data['his_lng_ovd_day']=test_data['his_lng_ovd_day'].astype('int')
test_data['his_lng_ovd_day_cls']=0
test_data['his_lng_ovd_day_cls'][test_data['his_lng_ovd_day']>0]=1
test_data['his_lng_ovd_day_cls'][test_data['his_lng_ovd_day']<=0]=0

train_data.to_csv('train_tag.csv',index=False)
test_data.to_csv('test_tag.csv',index=False)