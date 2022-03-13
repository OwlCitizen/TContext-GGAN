#dataset.py
import warnings
warnings.filterwarnings("ignore")
import time as timem
import pandas as pd
import numpy as np
import os
import math
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import torch.utils.data as Data
from utils import nanMask, maxminnorm
from concurrent.futures.thread import ThreadPoolExecutor
from concurrent.futures._base import as_completed
from cols import lab, medi, inputs
from multithreading import MyThread

folds_num=5

def lower_decay(tlist, max_length):
    decay_mask=[]
    for b in range(len(tlist)):
        t=tlist[b].clone().detach()
        t_=t.clone().detach()
        for i in range(1,len(t_)):
            t_[i]=t[i-1]
        gap=t-t_
        decay=1.0/torch.log(math.e+gap*1.0)
        padding=torch.zeros(max_length-len(decay))
        decay=torch.cat([decay,padding],dim=0)
        if b==0:
            decay_mask=torch.unsqueeze(decay,dim=0)
        else:
            decay_mask=torch.cat([decay_mask,torch.unsqueeze(decay,dim=0)],dim=0)
    return torch.unsqueeze(decay_mask,dim=-1)

def lower_decay_(t):#, max_length):
    t_=t.clone().detach()
    for i in range(1,len(t_)):
        t_[i]=t[i-1]
    gap=t-t_
    decay=1.0/torch.log(math.e+gap*1.0)
    return decay

def load_data(i,folds_info,path):
    print('loading fold '+str(i)+' data ......')
    fold_info=folds_info.loc[folds_info.fold==i]
    fold_labels=fold_info.label.values
    pbar=tqdm(fold_info.index)
    fold_data=[]
    fold_decay=[]
    fold_time=[]
    fold_mask=[]
    fold_lengths=[]
    fold_id=[]
    pbar.set_description('loading fold '+str(i))
    for p in pbar:
        frame=pd.read_csv(path+str(p)+'.csv', index_col=['hourint'])
        lab_mask=nanMask(frame.loc[:,lab].values)
        enc_lab=maxminnorm(frame.loc[:,lab])
        enc_lab.fillna(0,inplace=True)
        enc_inputs=maxminnorm(frame.loc[:,inputs])
        enc_med=frame.loc[:,medi]
        frame=pd.concat([enc_lab,enc_inputs,enc_med],axis=1)
        fold_decay.append(lower_decay_(torch.tensor(frame.index)))
        fold_time.append(torch.tensor(frame.index))
        fold_data.append(torch.tensor(frame.values))
        fold_mask.append(lab_mask)
        fold_lengths.append(frame.shape[0])
        fold_id.append(p)
    return fold_labels,fold_data,fold_decay,fold_time,fold_mask,fold_lengths,fold_id

def load_data_(i,folds_info,path,meanpadding=False):
    print('loading fold '+str(i)+' data ......')
    fold_info=folds_info.loc[folds_info.fold==i]
    fold_labels=fold_info.label.values
    pbar=tqdm(fold_info.index)
    fold_data=[]
    fold_time=[]
    fold_mask=[]
    fold_lengths=[]
    fold_id=[]
    pbar.set_description('loading fold '+str(i))
    for p in pbar:
        frame=pd.read_csv(path+str(p)+'.csv', index_col=['hourint'])
        lab_mask=nanMask(frame.loc[:,lab].values)
        enc_lab=maxminnorm(frame.loc[:,lab])
        enc_lab.fillna(0,inplace=True)
        enc_inputs=maxminnorm(frame.loc[:,inputs])
        enc_med=frame.loc[:,medi]
        frame=pd.concat([enc_lab,enc_inputs,enc_med],axis=1)
        padding=pd.DataFrame(np.zeros([1,frame.shape[1]]),index=[48],columns=frame.columns)
        frame=pd.concat([frame,padding],axis=0)
        fold_time.append(torch.tensor(frame.index))
        fold_data.append(torch.tensor(frame.values))
        fold_mask.append(lab_mask)
        fold_lengths.append(frame.shape[0])
        fold_id.append(p)
    return fold_labels,fold_data,fold_time,fold_mask,fold_lengths,fold_id

class HFDataset(torch.utils.data.Dataset):
    def __init__(self,task,info_name,wp=1.0):
        path='//data//xuyuyang//Graduate//'+task+'//'
        start=timem.time()
        folds_info=pd.read_csv(path+'data_info//'+info_name+'.csv',index_col='Unnamed: 0')
        self.wp=wp
        self.folds=[]
        self.folds_labels=[]
        self.folds_decays=[]
        self.folds_times=[]
        self.folds_mask=[]
        self.folds_lengths=[]
        self.folds_id=[]
        executor = ThreadPoolExecutor(max_workers=5)
        all_task=[]
        for i in range(folds_num):
            task = executor.submit(load_data,i,folds_info,path)
            all_task.append(task)
        for future in as_completed(all_task):
            (fold_labels,fold_data,fold_decay,fold_time,fold_mask,fold_lengths,fold_id) = future.result()
            self.folds_labels.append(fold_labels)
            self.folds.append(fold_data)
            self.folds_decays.append(fold_decay)
            self.folds_times.append(fold_time)
            self.folds_mask.append(fold_mask)
            self.folds_lengths.append(fold_lengths)
            self.folds_id.append(fold_id)
        self.train_label=[]#train_info.label
        self.val_label=[]
        self.test_label=[]#test_info.label
        self.train_set=[]
        self.val_set=[]
        self.test_set=[]
        self.train_decay=[]
        self.val_decay=[]
        self.test_decay=[]
        self.train_time=[]
        self.val_time=[]
        self.test_time=[]
        self.train_mask=[]
        self.val_mask=[]
        self.test_mask=[]
        self.train_lengths=[]
        self.val_lengths=[]
        self.test_lengths=[]
        self.train_id=[]
        self.test_id=[]
        self.val_id=[]
        self.train_w=[]
        self.mode=''
        end=timem.time()
        for i in range(len(self.folds)):
            assert len(self.folds[i])==len(self.folds_labels[i])
            assert len(self.folds[i])==len(self.folds_times[i])
            assert len(self.folds[i])==len(self.folds_mask[i])
            assert len(self.folds[i])==len(self.folds_lengths[i])
            assert len(self.folds[i])==len(self.folds_id[i])
            assert len(self.folds[i])==len(self.folds_decays[i])
        print('Data loading finish!')
        print('Takes: '+str(int((end-start)/60))+'min')
    def initfold(self,seed):
        start=timem.time()
        self.train_label=[]#train_info.label
        self.val_label=[]
        self.test_label=[]#test_info.label
        self.train_set=[]
        self.val_set=[]
        self.test_set=[]
        self.train_decay=[]
        self.val_decay=[]
        self.test_decay=[]
        self.train_time=[]
        self.test_time=[]
        self.val_time=[]
        self.train_mask=[]
        self.val_mask=[]
        self.test_mask=[]
        self.train_lengths=[]
        self.val_lengths=[]
        self.test_lengths=[]
        self.train_id=[]
        self.test_id=[]
        self.val_id=[]
        self.train_w=[]
        train_folds=[(seed%folds_num),((seed+1)%folds_num),((seed+2)%folds_num)]
        val_fold=(seed+3)%folds_num
        test_fold=(seed+4)%folds_num
        self.val_set=self.folds[val_fold]
        self.val_label=self.folds_labels[val_fold]
        self.val_decay=self.folds_decays[val_fold]
        self.val_time=self.folds_times[val_fold]
        self.val_mask=self.folds_mask[val_fold]
        self.val_lengths=self.folds_lengths[val_fold]
        self.val_id=self.folds_id[val_fold]
        self.test_set=self.folds[test_fold]
        self.test_label=self.folds_labels[test_fold]
        self.test_decay=self.folds_decays[test_fold]
        self.test_time=self.folds_times[test_fold]
        self.test_mask=self.folds_mask[test_fold]
        self.test_lengths=self.folds_lengths[test_fold]
        self.test_id=self.folds_id[test_fold]
        self.train_label=np.hstack([np.hstack([self.folds_labels[train_folds[0]],self.folds_labels[train_folds[1]]]),self.folds_labels[train_folds[2]]])
        #m=len(self.train_label)*1.0/sum(self.train_label)
        #self.train_w=torch.tensor(np.array([1.0/m,(1-1.0/m)*self.wp]), dtype=torch.float32)
        self.train_w=torch.tensor(np.array([1/(len(self.train_label)-sum(self.train_label)),1/sum(self.train_label)]), dtype=torch.float32)
        self.train_set=self.folds[train_folds[0]]+self.folds[train_folds[1]]+self.folds[train_folds[2]]
        self.train_time=self.folds_times[train_folds[0]]+self.folds_times[train_folds[1]]+self.folds_times[train_folds[2]]
        self.train_decay=self.folds_decays[train_folds[0]]+self.folds_decays[train_folds[1]]+self.folds_decays[train_folds[2]]
        self.train_mask=self.folds_mask[train_folds[0]]+self.folds_mask[train_folds[1]]+self.folds_mask[train_folds[2]]
        self.train_lengths=self.folds_lengths[train_folds[0]]+self.folds_lengths[train_folds[1]]+self.folds_lengths[train_folds[2]]
        self.train_id=self.folds_id[train_folds[0]]+self.folds_id[train_folds[1]]+self.folds_id[train_folds[2]]
        assert len(self.train_set)==len(self.train_label)
        assert len(self.train_set)==len(self.train_time)
        assert len(self.train_set)==len(self.train_mask)
        assert len(self.train_set)==len(self.train_lengths)
        assert len(self.train_set)==len(self.train_id)
        assert len(self.train_set)==len(self.train_decay)
        assert len(self.test_set)==len(self.test_label)
        assert len(self.test_set)==len(self.test_time)
        assert len(self.test_set)==len(self.test_mask)
        assert len(self.test_set)==len(self.test_lengths)
        assert len(self.test_set)==len(self.test_id)
        assert len(self.test_set)==len(self.test_decay)
        assert len(self.val_set)==len(self.val_label)
        assert len(self.val_set)==len(self.val_time)
        assert len(self.val_set)==len(self.val_mask)
        assert len(self.val_set)==len(self.val_lengths)
        assert len(self.val_set)==len(self.val_id)
        assert len(self.val_set)==len(self.val_decay)
        end=timem.time()
        print('Data processing finish!')
        print('Takes: '+str(int((end-start)/60))+'min')
    def initmode(self,mode):
        self.mode=mode
        if mode!='train' and mode!='test' and mode!='val':
            raise NotImplementedError
    def __getitem__(self, index):
        if len(self.train_set)==0 or len(self.test_set)==0 or len(self.val_set)==0:
            raise NotImplementedError
        if self.mode=='train':
            data = self.train_set[index]
            time=self.train_time[index]
            label=self.train_label[index]
            mask=self.train_mask[index]
            length=self.train_lengths[index]
            pid=self.train_id[index]
            decay=self.train_decay[index]
        elif self.mode=='val':
            data=self.val_set[index]
            time=self.val_time[index]
            label=self.val_label[index]
            mask=self.val_mask[index]
            length=self.val_lengths[index]
            pid=self.val_id[index]
            decay=self.val_decay[index]
        elif self.mode=='test':
            data=self.test_set[index]
            time=self.test_time[index]
            label=self.test_label[index]
            mask=self.test_mask[index]
            length=self.test_lengths[index]
            pid=self.test_id[index]
            decay=self.test_decay[index]
        else:
            raise NotImplementedError
        return data, decay, time, label, mask, length, pid
    def __len__(self):
        if len(self.train_set)==0 or len(self.test_set)==0 or len(self.val_set)==0:
            raise NotImplementedError
        if self.mode=='train':
            return len(self.train_set)
        elif self.mode=='test':
            return len(self.test_set)
        elif self.mode=='val':
            return len(self.val_set)
        else:
            raise NotImplementedError
