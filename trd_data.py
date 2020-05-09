import pandas as pd
import numpy as np

path_tag=['data/train_tag.csv','data/test_tag.csv']
path_trd=['train_data/训练数据集_trd.csv','评分数据集_trd_b.csv']
for i in range(len(path_tag)):
    trd_train_data=pd.read_csv(path_trd[i])
    train_data=pd.read_csv(path_tag[i])
    train_id=train_data['id']
    trd_id=trd_train_data['id'].drop_duplicates(keep='first')
    id_sum=pd.merge(train_id,trd_id,on='id')
    new_data=pd.DataFrame()
    new_data['id']=id_sum['id']
    new_data=new_data.set_index('id')
    trd_train_data['date'] = pd.to_datetime(trd_train_data['trx_tm'], format='%Y/%m/%d')
    trd_train_data['month']=trd_train_data['date'].dt.month
    trd_train_data['hour']=trd_train_data['date'].dt.hour
    trd_train_data['hour']=trd_train_data['hour'].astype('int')
    #时间转换成4类
    trd_train_data['midnight']=0
    trd_train_data['morning']=0
    trd_train_data['afternoon']=0
    trd_train_data['night']=0
    trd_train_data['midnight'][(trd_train_data['hour']<6) & (trd_train_data['hour']>=0)]=1
    trd_train_data['morning'][(trd_train_data['hour']<12) & (trd_train_data['hour']>=6)]=1
    trd_train_data['afternoon'][(trd_train_data['hour']<18) & (trd_train_data['hour']>=12)]=1
    trd_train_data['night'][(trd_train_data['hour']<24) & (trd_train_data['hour']>=18)]=1
    trd_train_data.drop(columns='hour',inplace=True)
    #支付方式，有三种，统计个数
    new_data['Dir_A']=0
    new_data['Dir_B']=0
    new_data['Dir_C']=0
    #一类代码，有三级，分三维统计个数
    new_data['Trx_Cod1_Cd_1']=0
    new_data['Trx_Cod1_Cd_2']=0
    new_data['Trx_Cod1_Cd_3']=0
    #二类代码，用一维，取众数
    if i==0:
        code2=trd_train_data['Trx_Cod2_Cd'].unique()
    for j in code2:
        col_name='Trx_Cod2_Cd_'+str(j)
        new_data[col_name]=0
    #输入、支出分别的最大值、最小值、均值、次数
    new_data['B_amt_min']=0
    new_data['B_amt_max']=0
    new_data['B_amt_mean']=0
    new_data['B_amt_count']=0

    new_data['C_amt_min']=0
    new_data['C_amt_max']=0
    new_data['C_amt_mean']=0
    new_data['C_amt_count']=0

    new_data['midnight']=0
    new_data['morning']=0
    new_data['afternoon']=0
    new_data['night']=0

    new_data['month_1']=0
    new_data['month_2']=0
    new_data['month_3']=0
    new_data['month_4']=0
    new_data['month_5']=0
    new_data['month_6']=0
    new_data['month_7']=0
    new_data['month_8']=0
    new_data['month_9']=0
    new_data['month_10']=0
    new_data['month_11']=0
    new_data['month_12']=0
    trd_train_data=trd_train_data.groupby('id')
    for index,row in trd_train_data:
        temp=trd_train_data.get_group(index)
        if index in new_data.index:
            #交易方式统计
            if temp[temp['Dat_Flg3_Cd']=='A'].shape[0]!=0:
                new_data['Dir_A'].loc[index]=temp[temp['Dat_Flg3_Cd']=='A'].shape[0]
            if temp[temp['Dat_Flg3_Cd']=='B'].shape[0]!=0:
                new_data['Dir_B'].loc[index]=temp[temp['Dat_Flg3_Cd']=='B'].shape[0]
            if temp[temp['Dat_Flg3_Cd']=='C'].shape[0]!=0:
                new_data['Dir_C'].loc[index]=temp[temp['Dat_Flg3_Cd']=='C'].shape[0]

            #一级代码统计
            if temp[temp['Trx_Cod1_Cd']=='1'].shape[0]!=0:
                new_data['Trx_Cod1_Cd_1'].loc[index]=temp[temp['Trx_Cod1_Cd']=='1'].shape[0]
            if temp[temp['Trx_Cod1_Cd']=='2'].shape[0]!=0:
                new_data['Trx_Cod1_Cd_2'].loc[index]=temp[temp['Trx_Cod1_Cd']=='2'].shape[0]
            if temp[temp['Trx_Cod1_Cd']=='3'].shape[0]!=0:
                new_data['Trx_Cod1_Cd_3'].loc[index]=temp[temp['Trx_Cod1_Cd']=='3'].shape[0]

            #二级代码取众数
            for k in code2:
                col_name='Trx_Cod2_Cd_'+str(k)
                if temp[temp['Trx_Cod2_Cd']==k].shape[0]!=0:
                    new_data[col_name].loc[index]=temp[temp['Trx_Cod2_Cd']==k].shape[0]

            #收入支出的信息统计
            info_temp=temp[temp['Dat_Flg1_Cd']=='B']['cny_trx_amt']
            if info_temp.shape[0]!=0:
                new_data['B_amt_min'].loc[index]=info_temp.min()
                new_data['B_amt_max'].loc[index]=info_temp.max()
                new_data['B_amt_mean'].loc[index]=info_temp.mean()
                new_data['B_amt_count'].loc[index]=info_temp.shape[0]

            info_temp=temp[temp['Dat_Flg1_Cd']=='C']['cny_trx_amt']
            if info_temp.shape[0]!=0:
                new_data['C_amt_min'].loc[index]=info_temp.min()
                new_data['C_amt_max'].loc[index]=info_temp.max()
                new_data['C_amt_mean'].loc[index]=info_temp.mean()
                new_data['C_amt_count'].loc[index]=info_temp.shape[0]

            #交易时间段统计
            if temp[temp['midnight']==1].shape[0]!=0:
                new_data['midnight'].loc[index]=temp[temp['midnight']==1].shape[0]
            if temp[temp['morning']==1].shape[0]!=0:
                new_data['morning'].loc[index]=temp[temp['morning']==1].shape[0]
            if temp[temp['afternoon']==1].shape[0]!=0:
                new_data['afternoon'].loc[index]=temp[temp['afternoon']==1].shape[0]
            if temp[temp['night']==1].shape[0]!=0:
                new_data['night'].loc[index]=temp[temp['night']==1].shape[0]
            
            #交易月份统计
            if temp[temp['month']==1].shape[0]!=0:
                new_data['month_1'].loc[index]=temp[temp['month']==1].shape[0]
            if temp[temp['month']==2].shape[0]!=0:
                new_data['month_2'].loc[index]=temp[temp['month']==2].shape[0]
            if temp[temp['month']==3].shape[0]!=0:
                new_data['month_3'].loc[index]=temp[temp['month']==3].shape[0]
            if temp[temp['month']==4].shape[0]!=0:
                new_data['month_4'].loc[index]=temp[temp['month']==4].shape[0]
            if temp[temp['month']==5].shape[0]!=0:
                new_data['month_5'].loc[index]=temp[temp['month']==5].shape[0]
            if temp[temp['month']==6].shape[0]!=0:
                new_data['month_6'].loc[index]=temp[temp['month']==6].shape[0]
            if temp[temp['month']==7].shape[0]!=0:
                new_data['month_7'].loc[index]=temp[temp['month']==7].shape[0]
            if temp[temp['month']==8].shape[0]!=0:
                new_data['month_8'].loc[index]=temp[temp['month']==8].shape[0]
            if temp[temp['month']==9].shape[0]!=0:
                new_data['month_9'].loc[index]=temp[temp['month']==9].shape[0]
            if temp[temp['month']==10].shape[0]!=0:
                new_data['month_10'].loc[index]=temp[temp['month']==10].shape[0]
            if temp[temp['month']==11].shape[0]!=0:
                new_data['month_11'].loc[index]=temp[temp['month']==11].shape[0]
            if temp[temp['month']==12].shape[0]!=0:
                new_data['month_12'].loc[index]=temp[temp['month']==12].shape[0]
    #二级代码One-hot
    #temp=pd.get_dummies(new_data['Trx_Cod2_Cd'],prefix="code2").astype('float')
    #new_data[temp.columns]=temp
    #new_data.drop(columns='Trx_Cod2_Cd',inplace=True)
    
    print(new_data.isnull().sum())
    if i==0:
        new_data.to_csv('data/train_trd.csv')
    elif i==1:
        new_data.to_csv('data/test_trd.csv')