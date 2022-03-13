#utils.py
import numpy as np
import torch
from sklearn import preprocessing
import math

def maxminnorm(frame):
    min_max_scaler = preprocessing.MinMaxScaler()
    for i in range(frame.shape[1]):
        frame.iloc[:,i]=min_max_scaler.fit_transform(frame.iloc[:,i].values.reshape(-1,1))
    return frame

def upper(m):
    nm=torch.zeros(m.shape)
    if len(m.shape)>1:
        for i in range(m.shape[0]-1):
            nm[i,:]=m[i+1,:]
    if m.is_cuda:
        nm=nm.cuda()
    return nm

def upper_(m):
    eye=torch.eye(m.shape[-2])
    if m.is_cuda:
        eye=eye.cuda()
    return torch.matmul(upper(eye),m)

def lower(m):
    nm=torch.zeros(m.shape)
    if len(m.shape)>1:
        for i in range(1,m.shape[0]):
            nm[-i,:]=m[-i-1,:]
    if m.is_cuda:
        nm=nm.cuda()
    return nm

def lower_(m,length):
    eye=torch.eye(m.shape[-2])
    batch_size=m.shape[0]
    if m.is_cuda:
        eye=eye.cuda()
    result=torch.matmul(lower(eye),m)
    for b in range(batch_size):
        blength=length[b]
        for r in range(blength,max(length)):
            result[b][r]=0
    return torch.matmul(lower(eye),m)

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
        decay_mask.append(torch.unsqueeze(decay, dim = 0))
    torch.cat(decay_mask, dim = 0)
    return torch.unsqueeze(decay_mask,dim=-1)

def upper_decay(tlist, max_length):
    decay_mask=[]
    for b in range(len(tlist)):
        t=tlist[b]
        t_=t.clone().detach()
        for i in range(0,len(t_)-1):
            t_[i]=t[i+1]
        gap=t_-t
        decay=1.0/torch.log(math.e+gap*1.0)
        padding=torch.zeros(max_length-len(decay))
        decay=torch.cat([decay,padding],dim=0)
        if b==0:
            decay_mask=torch.unsqueeze(decay,dim=0)
        else:
            decay_mask=torch.cat([decay_mask,torch.unsqueeze(decay,dim=0)],dim=0)
    return torch.unsqueeze(decay_mask,dim=-1)

def nanMask(m):
    mask=torch.ones(m.shape)
    if len(m.shape)==3:
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                for k in range(m.shape[2]):
                    if np.isnan(m[i][j]):
                        mask[i][j][k]=0
    else:
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if np.isnan(m[i][j]):
                    mask[i][j]=0
    return mask

def zeroMask(m):
    mask=torch.zeros(m.shape)
    if len(m.shape)==3:
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                for k in range(m.shape[2]):
                    if m[i][j][k]!=0:
                        mask[i][j][k]=1
    else:
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if m[i][j]!=0:
                    mask[i][j]=1
    if m.is_cuda:
        mask=mask.cuda()
    return mask

def visit_collate_fn(batch):
    data, time, label, mask, lengths, pid = zip(*batch)
    #num_features = batch_seq[0].shape[1]
    data_features=data[0].shape[1]
    mask_features=mask[0].shape[1]
    #seq_lengths = list(map(lambda patient_tensor: patient_tensor.shape[0], batch_seq))
    seq_lengths=lengths
    max_length = max(seq_lengths)
    
    sorted_indices, sorted_lengths = zip(*sorted(enumerate(seq_lengths), key=lambda x: x[1], reverse=True))
    sorted_padded_data = []
    sorted_padded_mask = []
    sorted_labels = []
    sorted_time=[]
    sorted_pid=[]
    
    for i in sorted_indices:
        length = data[i].shape[0]
        if length < max_length:
            padded_data=np.concatenate((data[i].numpy(), np.zeros((max_length - length, data_features), dtype=np.float32)), axis=0)
            padded_mask=np.concatenate((mask[i].numpy(), np.zeros((max_length - length, mask_features), dtype=np.float32)), axis=0)
        else:
            padded_data=data[i].numpy()
            padded_mask=mask[i].numpy()
        sorted_padded_data.append(padded_data)
        sorted_padded_mask.append(padded_mask)
        sorted_labels.append(label[i])
        sorted_time.append(time[i])
        sorted_pid.append(pid[i])
        
    data_tensor = np.stack(sorted_padded_data, axis=0)
    mask_tensor = np.stack(sorted_padded_mask, axis=0)
    label_tensor = torch.LongTensor(sorted_labels)
    
    return torch.from_numpy(data_tensor), list(sorted_time), label_tensor, torch.from_numpy(mask_tensor), list(sorted_lengths), list(sorted_pid)

def visit_collate_fn_(batch):
    data, decay, time, label, mask, lengths, pid = zip(*batch)
    #num_features = batch_seq[0].shape[1]
    data_features=data[0].shape[1]
    mask_features=mask[0].shape[1]
    #seq_lengths = list(map(lambda patient_tensor: patient_tensor.shape[0], batch_seq))
    seq_lengths=lengths
    max_length = max(seq_lengths)
    
    sorted_indices, sorted_lengths = zip(*sorted(enumerate(seq_lengths), key=lambda x: x[1], reverse=True))
    sorted_padded_data = []
    sorted_padded_mask = []
    sorted_padded_decay = []
    sorted_labels = []
    sorted_time=[]
    sorted_pid=[]
    
    for i in sorted_indices:
        length = data[i].shape[0]
        if length < max_length:
            padded_data=np.concatenate((data[i].numpy(), np.zeros((max_length - length, data_features), dtype=np.float32)), axis=0)
            padded_mask=np.concatenate((mask[i].numpy(), np.zeros((max_length - length, mask_features), dtype=np.float32)), axis=0)
            padded_decay=np.concatenate((decay[i].numpy(), np.zeros((max_length - length), dtype = np.float32)), axis = 0)
        else:
            padded_data=data[i].numpy()
            padded_mask=mask[i].numpy()
            padded_decay=decay[i].numpy()
        sorted_padded_data.append(padded_data)
        sorted_padded_mask.append(padded_mask)
        sorted_padded_decay.append(padded_decay)
        sorted_labels.append(label[i])
        sorted_time.append(time[i])
        sorted_pid.append(pid[i])
        
    data_tensor = np.stack(sorted_padded_data, axis=0)
    mask_tensor = np.stack(sorted_padded_mask, axis=0)
    decay_tensor = np.stack(sorted_padded_decay, axis=0)
    label_tensor = torch.LongTensor(sorted_labels)
    
    return torch.from_numpy(data_tensor), torch.from_numpy(decay_tensor), list(sorted_time), label_tensor, torch.from_numpy(mask_tensor), list(sorted_lengths), list(sorted_pid)
