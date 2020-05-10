import pandas as pd
import numpy as np

train_data=pd.read_csv('data/clean_train_tag.csv')
test_data=pd.read_csv('data/clean_test_tag.csv')
#性别转化为0 1
train_data['gdr_cd'][train_data['gdr_cd']=='F']=0
train_data['gdr_cd'][train_data['gdr_cd']=='M']=1

test_data['gdr_cd'][test_data['gdr_cd']=='F']=0
test_data['gdr_cd'][test_data['gdr_cd']=='M']=1
#付款方式one hot
temp=pd.get_dummies(train_data['atdd_type'],prefix='atdd_type').astype('int')
train_data[temp.columns]=temp
train_data.drop(columns='atdd_type',inplace=True)

temp=pd.get_dummies(test_data['atdd_type'],prefix='atdd_type').astype('int')
test_data[temp.columns]=temp
test_data.drop(columns='atdd_type',inplace=True)

#学历、教育程度onehot
temp=pd.get_dummies(train_data['deg_cd'],prefix='deg_cd').astype('int')
train_data[temp.columns]=temp
train_data.drop(columns='deg_cd',inplace=True)

temp=pd.get_dummies(test_data['deg_cd'],prefix='deg_cd').astype('int')
test_data[temp.columns]=temp
test_data.drop(columns='deg_cd',inplace=True)

temp=pd.get_dummies(train_data['edu_deg_cd'],prefix='edu_deg_cd').astype('int')
train_data[temp.columns]=temp
train_data.drop(columns='edu_deg_cd',inplace=True)

temp=pd.get_dummies(test_data['edu_deg_cd'],prefix='edu_deg_cd').astype('int')
test_data[temp.columns]=temp
test_data.drop(columns='edu_deg_cd',inplace=True)


#'mrg_situ_cd'有6类，onehot
temp=pd.get_dummies(train_data['mrg_situ_cd'],prefix="mrg_situ_cd").astype('int')
train_data[temp.columns]=temp
train_data.drop(columns='mrg_situ_cd',inplace=True)

temp=pd.get_dummies(test_data['mrg_situ_cd'],prefix="mrg_situ_cd").astype('int')
test_data[temp.columns]=temp
test_data.drop(columns='mrg_situ_cd',inplace=True)
#acdm_deg_cd有8类,onehot
temp=pd.get_dummies(train_data['acdm_deg_cd'],prefix="acdm_deg_cd").astype('int')
train_data[temp.columns]=temp
train_data.drop(columns='acdm_deg_cd',inplace=True)

temp=pd.get_dummies(test_data['acdm_deg_cd'],prefix="acdm_deg_cd").astype('int')
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

train_data['age_new']=0
train_data['age_new'][train_data['age']>60]=3
train_data['age_new'][(train_data['age']<=60)&(train_data['age']>50)]=2
train_data['age_new'][(train_data['age']<=50)&(train_data['age']>40)]=1
train_data['age_new'][(train_data['age']<=40)&(train_data['age']>30)]=0

test_data['age_new']=0
test_data['age_new'][test_data['age']>60]=3
test_data['age_new'][(test_data['age']<=60)&(test_data['age']>50)]=2
test_data['age_new'][(test_data['age']<=50)&(test_data['age']>40)]=1
test_data['age_new'][(test_data['age']<=40)&(test_data['age']>30)]=0

#工作年限分类
train_data['job_year'] = train_data['job_year'].astype('int')
test_data['job_year'] = test_data['job_year'].astype('int')

train_data['job_new']=0
train_data['job_new'][train_data['job_year']>30]=4
train_data['job_new'][(train_data['job_year']<=30)&(train_data['job_year']>20)]=3
train_data['job_new'][(train_data['job_year']<=20)&(train_data['job_year']>10)]=2
train_data['job_new'][(train_data['job_year']<=10)&(train_data['job_year']>0)]=1


test_data['job_new']=0
test_data['job_new'][test_data['job_year']>30]=4
test_data['job_new'][(test_data['job_year']<=30)&(test_data['job_year']>20)]=3
test_data['job_new'][(test_data['job_year']<=20)&(test_data['job_year']>10)]=2
test_data['job_new'][(test_data['job_year']<=10)&(test_data['job_year']>0)]=1

##借记卡按-1，1-5年，5-10年，10年以上分类
train_data['cur_debit_min_opn_dt_cnt'] = train_data['cur_debit_min_opn_dt_cnt'].astype('int')
test_data['cur_debit_min_opn_dt_cnt'] = test_data['cur_debit_min_opn_dt_cnt'].astype('int')

train_data['cur_debit_min_opn_dt_cnt_new']=-1
train_data['cur_debit_min_opn_dt_cnt_new'][(train_data['cur_debit_min_opn_dt_cnt']>-1)&(train_data['cur_debit_min_opn_dt_cnt']<=1825)]=1
train_data['cur_debit_min_opn_dt_cnt_new'][(train_data['cur_debit_min_opn_dt_cnt']>1825)&(train_data['cur_debit_min_opn_dt_cnt']<=3650)]=2
train_data['cur_debit_min_opn_dt_cnt_new'][(train_data['cur_debit_min_opn_dt_cnt']>3650)]=3


test_data['cur_debit_min_opn_dt_cnt_new']=-1
test_data['cur_debit_min_opn_dt_cnt_new'][(test_data['cur_debit_min_opn_dt_cnt']>-1)&(test_data['cur_debit_min_opn_dt_cnt']<=1825)]=1
test_data['cur_debit_min_opn_dt_cnt_new'][(test_data['cur_debit_min_opn_dt_cnt']>1825)&(test_data['cur_debit_min_opn_dt_cnt']<=3650)]=2
test_data['cur_debit_min_opn_dt_cnt_new'][(test_data['cur_debit_min_opn_dt_cnt']>3650)]=3

#按-1，1-3年，3-5年，5-10年，10年以上分类
train_data['cur_credit_min_opn_dt_cnt'] = train_data['cur_credit_min_opn_dt_cnt'].astype('int')
test_data['cur_credit_min_opn_dt_cnt'] = test_data['cur_credit_min_opn_dt_cnt'].astype('int')

train_data['cur_credit_min_opn_dt_cnt_new']=-1
train_data['cur_credit_min_opn_dt_cnt_new'][(train_data['cur_credit_min_opn_dt_cnt']>-1)&(train_data['cur_credit_min_opn_dt_cnt']<=1095)]=1
train_data['cur_credit_min_opn_dt_cnt_new'][(train_data['cur_credit_min_opn_dt_cnt']>1095)&(train_data['cur_credit_min_opn_dt_cnt']<=1825)]=2
train_data['cur_credit_min_opn_dt_cnt_new'][(train_data['cur_credit_min_opn_dt_cnt']>1825)&(train_data['cur_credit_min_opn_dt_cnt']<=3650)]=3
train_data['cur_credit_min_opn_dt_cnt_new'][(train_data['cur_credit_min_opn_dt_cnt']>3650)]=4

test_data['cur_credit_min_opn_dt_cnt_new']=-1
test_data['cur_credit_min_opn_dt_cnt_new'][(test_data['cur_credit_min_opn_dt_cnt']>-1)&(test_data['cur_credit_min_opn_dt_cnt']<=1095)]=1
test_data['cur_credit_min_opn_dt_cnt_new'][(test_data['cur_credit_min_opn_dt_cnt']>1095)&(test_data['cur_credit_min_opn_dt_cnt']<=1825)]=2
test_data['cur_credit_min_opn_dt_cnt_new'][(test_data['cur_credit_min_opn_dt_cnt']>1825)&(test_data['cur_credit_min_opn_dt_cnt']<=3650)]=1
test_data['cur_credit_min_opn_dt_cnt_new'][(test_data['cur_credit_min_opn_dt_cnt']>3650)]=4

#按不逾期，逾期一笔，逾期一笔以上分类计算
train_data['ovd_30d_loan_tot_cnt'] = train_data['ovd_30d_loan_tot_cnt'].astype('int')
test_data['ovd_30d_loan_tot_cnt'] = test_data['ovd_30d_loan_tot_cnt'].astype('int')

train_data['ovd_30d_loan_tot_cnt_new']=0
train_data['ovd_30d_loan_tot_cnt_new'][train_data['ovd_30d_loan_tot_cnt']==1]=1
train_data['ovd_30d_loan_tot_cnt_new'][train_data['ovd_30d_loan_tot_cnt']>1]=2


test_data['ovd_30d_loan_tot_cnt_new']=0
test_data['ovd_30d_loan_tot_cnt_new'][test_data['ovd_30d_loan_tot_cnt']==1]=1
test_data['ovd_30d_loan_tot_cnt_new'][test_data['ovd_30d_loan_tot_cnt']>1]=2


train_data['cur_debit_min_opn_dt_cnt']=(train_data['cur_debit_min_opn_dt_cnt']-train_data['cur_debit_min_opn_dt_cnt'].min())/(train_data['cur_debit_min_opn_dt_cnt'].max()-train_data['cur_debit_min_opn_dt_cnt'].min())
train_data['cur_credit_min_opn_dt_cnt']=(train_data['cur_credit_min_opn_dt_cnt']-train_data['cur_credit_min_opn_dt_cnt'].min())/(train_data['cur_credit_min_opn_dt_cnt'].max()-train_data['cur_credit_min_opn_dt_cnt'].min())
train_data['his_lng_ovd_day']=(train_data['his_lng_ovd_day']-train_data['his_lng_ovd_day'].min())/(train_data['his_lng_ovd_day'].max()-train_data['his_lng_ovd_day'].min())
train_data['cur_debit_cnt']=(train_data['cur_debit_cnt']-train_data['cur_debit_cnt'].min())/(train_data['cur_debit_cnt'].max()-train_data['cur_debit_cnt'].min())
train_data['cur_credit_cnt']=(train_data['cur_credit_cnt']-train_data['cur_credit_cnt'].min())/(train_data['cur_credit_cnt'].max()-train_data['cur_credit_cnt'].min())
train_data['cur_cnt']=(train_data['cur_credit_cnt']-train_data['cur_cnt'].min())/(train_data['cur_cnt'].max()-train_data['cur_cnt'].min())
train_data['age']=(train_data['age']-train_data['age'].min())/(train_data['age'].max()-train_data['age'].min())

test_data['cur_debit_min_opn_dt_cnt']=(test_data['cur_debit_min_opn_dt_cnt']-test_data['cur_debit_min_opn_dt_cnt'].min())/(test_data['cur_debit_min_opn_dt_cnt'].max()-test_data['cur_debit_min_opn_dt_cnt'].min())
test_data['cur_credit_min_opn_dt_cnt']=(test_data['cur_credit_min_opn_dt_cnt']-test_data['cur_credit_min_opn_dt_cnt'].min())/(test_data['cur_credit_min_opn_dt_cnt'].max()-test_data['cur_credit_min_opn_dt_cnt'].min())
test_data['his_lng_ovd_day']=(test_data['his_lng_ovd_day']-test_data['his_lng_ovd_day'].min())/(test_data['his_lng_ovd_day'].max()-test_data['his_lng_ovd_day'].min())
test_data['cur_debit_cnt']=(test_data['cur_debit_cnt']-test_data['cur_debit_cnt'].min())/(test_data['cur_debit_cnt'].max()-test_data['cur_debit_cnt'].min())
test_data['cur_credit_cnt']=(test_data['cur_credit_cnt']-test_data['cur_credit_cnt'].min())/(test_data['cur_credit_cnt'].max()-test_data['cur_credit_cnt'].min())
test_data['cur_cnt']=(test_data['cur_cnt']-test_data['cur_cnt'].min())/(test_data['cur_cnt'].max()-test_data['cur_cnt'].min())
test_data['age']=(test_data['age']-test_data['age'].min())/(test_data['age'].max()-test_data['age'].min())
train_data.to_csv('data/train_tag.csv',index=False)
test_data.to_csv('data/test_tag.csv',index=False)