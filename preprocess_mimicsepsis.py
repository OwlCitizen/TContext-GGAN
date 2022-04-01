import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

data=pd.read_csv('.//sepsis48_hour_iv_all.csv',index_col='hadm_id')
patients=data.index.drop_duplicates()
pbar=tqdm(patients)
samples=[]
labels=[]
lengths=[]
fold_num=5
cols=list(data.columns)
#cols.remove('expire_flag')
cols.remove('final_expire_flag')
medi=['prednisolone'
	, 'methylprednisolone'
	, 'tobradex'
	, 'dexamethasone'
	, 'hydrocortisone']
inputs=['endoscopy'
	, 'tubation'
	, 'invasive'
	, 'dialysis']
lab=['arterial_bp_systolic'
	, 'arterial_bp_diastolic'
	, 'gcs_verbal_response'
	, 'gcs_motor_response'
	, 'gcs_eye_opening'
	, 'albumin'
	, 'heart_rate'
	, 'respiratory_rate'
	, 'spo2'
	, 'hemoglobin'
	, 'body_temperature'
	, 'platelet'
	, 'potassium'
	, 'bun']

def str2date(date):
    date=date.split(' ')[0].split('/')
    yy=int(date[0])
    mm=int(date[1])
    dd=int(date[2])
    total=yy*372+mm*31+dd
    return total

def maxminnorm(df):
    min=df.min(axis=0)
    max=df.max(axis=0)
    return (df-min)/(max-min)

for p in pbar:
    temp=data.loc[p]
    if len(temp.shape)==2:
        #if temp.min().time+30<=temp.max().time:
        samples.append(p)
        labels.append(temp.max().final_expire_flag)
        temp=temp.reset_index()
        temp=temp.loc[:,cols]
        #temp=temp.loc[temp.hourint<temp.min().hourint+48]
        temp=temp.sort_values(by='hourint',ascending=True)
        temp=temp.set_index(['hourint'])
        lengths.append(temp.shape[0])
        temp.to_csv('.//sepsis48_hour_iv_all//'+str(p)+'.csv')
            
print('po size:'+str(sum(labels)))
print('all size:'+str(len(labels)))

data_sta=pd.DataFrame(labels,index=samples,columns=['labels'])
data_sta['lengths']=lengths
data_sta.to_csv('.//sepsis48_hour_iv_all//data_info//data_info.csv')

po_info=data_sta.loc[data_sta.labels==1]
ne_info=data_sta.loc[data_sta.labels==0]

folds=[]

x_po_train,x_po_test,y_po_train,y_po_test = train_test_split(po_info.index,po_info.labels,test_size=0.3,random_state=5)
x_ne_train,x_ne_test,y_ne_train,y_ne_test = train_test_split(ne_info.index,ne_info.labels,test_size=0.3,random_state=5)
train=pd.concat([pd.DataFrame(y_po_train.values,columns=['label'],index=x_po_train),pd.DataFrame(y_ne_train.values,columns=['label'],index=x_ne_train)])
test=pd.concat([pd.DataFrame(y_po_test.values,columns=['label'],index=x_po_test),pd.DataFrame(y_ne_test.values,columns=['label'],index=x_ne_test)])

print('train_size:'+str(len(train)))
print('test_size:'+str(len(test)))
train.to_csv('.//sepsis48_hour_iv_all//data_info//train.csv')
test.to_csv('.//sepsis48_hour_iv_all//data_info//test.csv')

x_po_rest=po_info.index
y_po_rest=po_info.labels.values
x_ne_rest=ne_info.index
y_ne_rest=ne_info.labels.values

for i in range(fold_num-1):
    x_po_rest,x_po_fold,y_po_rest,y_po_fold = train_test_split(x_po_rest,y_po_rest,test_size=(1.0/(fold_num-i)),random_state=1)
    x_ne_rest,x_ne_fold,y_ne_rest,y_ne_fold = train_test_split(x_ne_rest,y_ne_rest,test_size=(1.0/(fold_num-i)),random_state=1)
    fold=pd.concat([pd.DataFrame(y_po_fold,columns=['label'],index=x_po_fold),pd.DataFrame(y_ne_fold,columns=['label'],index=x_ne_fold)])
    print('fold '+str(i)+' shape: '+str(fold.shape))
    fold.to_csv('.//sepsis48_hour_iv_all//data_info//fold'+str(i)+'_info.csv')
    folds.append(fold)
    
fold=pd.concat([pd.DataFrame(y_po_rest,columns=['label'],index=x_po_rest),pd.DataFrame(y_ne_rest,columns=['label'],index=x_ne_rest)])
print('fold '+str(i+1)+' shape: '+str(fold.shape))
fold.to_csv('.//sepsis48_hour_iv_all//data_info//fold'+str(fold_num-1)+'_info.csv')
folds.append(fold)

folds_info=[]
count=0
for f in folds:
    f.loc[:,'fold']=count
    if count==0:
        folds_info=f
    else:
        folds_info=pd.concat([folds_info,f],axis=0)
    count+=1
    
folds_info.to_csv('.//sepsis48_hour_iv_all//data_info//folds_info.csv')




print('finish!')