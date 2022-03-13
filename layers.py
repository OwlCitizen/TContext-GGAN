#layers.py
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
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
from tensorboardX import SummaryWriter
import copy
from sklearn import preprocessing
from utils import upper_, lower_, lower_decay, upper_decay
from cols import lab, medi, inputs

class SelfAttentionLayer(nn.Module):
    def __init__(self, dim_embd=len(lab+medi+inputs), dp_rate=0.5, qkv_diff=True, pass_v=True, out=False):
        super(SelfAttentionLayer, self).__init__()
        self.dp_rate=dp_rate
        self.dim_embd=dim_embd
        self.qkv_diff=qkv_diff
        self.pass_v=pass_v
        self.out=out
        
        self.Wq=nn.Sequential(
                nn.Linear(dim_embd, dim_embd, bias=False),
                nn.Dropout(p=dp_rate)
            )
        self.Wk=nn.Sequential(
                nn.Linear(dim_embd, dim_embd, bias=False),
                nn.Dropout(p=dp_rate)
            )
        self.Wv=nn.Sequential(
                nn.Linear(dim_embd, dim_embd, bias=False),
                nn.Dropout(p=dp_rate)
            )
        
    def reset_params(self):
        torch.nn.init.kaiming_normal_(self.Wq[0].weight)
        torch.nn.init.kaiming_normal_(self.Wk[0].weight)
        torch.nn.init.kaiming_normal_(self.Wv[0].weight)
        
    def forward(self, h_e,h_l,h_i,h_m,length):
        batch_size=len(h_e)
        assert batch_size == len(length)
        
        if self.qkv_diff:
            h_c=torch.cat([h_l,h_i,h_m],dim=1)
            Q=self.Wq(h_e)
            K=self.Wk(h_c)
            V=self.Wv(h_c)
            if self.pass_v:
                P=V
            else:
                P=K
            Z=torch.matmul(F.softmax(torch.matmul(Q,K.transpose(-1,-2))/math.sqrt(K.shape[-1]),dim=-1),V)
            '''
            for b in range(batch_size):
                blength=length[b]
                for r in range(blength,max(length)):
                    Z[b][r]=0
            '''
            for b, blength in enumerate(length):
                Z[b][blength:]=0
                
            h_e=h_e+Z
            h_l=h_l+P[:,:len(lab),:]
            h_i=h_i+P[:,len(lab):len(lab+inputs),:]
            h_m=h_m+P[:,len(lab+inputs):,:]
        else:
            h_c=torch.cat([h_e,h_l,h_i,h_m],dim=1)
            Q=self.Wq(h_c)
            K=self.Wk(h_c)
            V=self.Wv(h_c)
            
            Z=torch.matmul(F.softmax(torch.matmul(Q,K.transpose(-1,-2))/math.sqrt(K.shape[-1]),dim=-1),V)
            '''
            for b in range(batch_size):
                blength=length[b]
                for r in range(blength,max(length)):
                    Z[b][r]=0
            '''
            for b, blength in enumerate(length):
                Z[b][blength:]=0
            
            h_e=h_e+Z[:,:max(length),:]
            h_l=h_l+Z[:,max(length):max(length)+len(lab),:]
            h_i=h_i+Z[:,max(length)+len(lab):max(length)+len(lab+inputs),:]
            h_m=h_m+Z[:,max(length)+len(lab+inputs):,:]
        
        if not self.out:
            return h_e,h_l,h_i,h_m
        else:
            return h_e

class TGGANN(nn.Module):
    def __init__(self, dim_embd, loop_num, nodecay=False, basic=False):
        super(TGGANN, self).__init__()
        ########
        self.dim_embd=dim_embd
        self.loop_num=loop_num
        self.nodecay=nodecay
        self.basic = basic
        if self.basic:
            self.nodecay = True
        if not self.nodecay and not self.basic:
            #enc_decay
            self.W_decay_enc=nn.Linear(dim_embd, dim_embd)
            #med_decay
            self.W_decay_med=nn.Linear(dim_embd, dim_embd)
            #inputs_decay
            self.W_decay_inputs=nn.Linear(dim_embd, dim_embd)
        ##########
        self.vars_r_in=torch.nn.ParameterList()
        #weight 0
        weight_lab_r=nn.Parameter(torch.ones(dim_embd*2,dim_embd))
        self.vars_r_in.append(weight_lab_r)
        #weight 1
        weight_inputs_r=nn.Parameter(torch.ones(dim_embd*2,dim_embd))
        self.vars_r_in.append(weight_inputs_r)
        #weight 2
        weight_med_r=nn.Parameter(torch.ones(dim_embd*2,dim_embd))
        self.vars_r_in.append(weight_med_r)
        #weight 3
        weight_enc_r=nn.Parameter(torch.ones(dim_embd*2,dim_embd))
        self.vars_r_in.append(weight_enc_r)
        ##################
        self.vars_out=torch.nn.ParameterList()
        #weight 4
        weight_lab_r=nn.Parameter(torch.ones(dim_embd,dim_embd))
        self.vars_out.append(weight_lab_r)
        #weight 5
        weight_inputs_r=nn.Parameter(torch.ones(dim_embd,dim_embd))
        self.vars_out.append(weight_inputs_r)
        #weight 6
        weight_med_r=nn.Parameter(torch.ones(dim_embd,dim_embd))
        self.vars_out.append(weight_med_r)
        #weight 7
        weight_enc_r=nn.Parameter(torch.ones(dim_embd,dim_embd))
        self.vars_out.append(weight_enc_r)
        ##########
        self.vars_z_in=torch.nn.ParameterList()
        #weight 8
        weight_lab_z=nn.Parameter(torch.ones(2*dim_embd,2*dim_embd))
        self.vars_z_in.append(weight_lab_z)
        #weight 9
        weight_inputs_z=nn.Parameter(torch.ones(2*dim_embd,2*dim_embd))
        self.vars_z_in.append(weight_inputs_z)
        #weight 10
        weight_med_z=nn.Parameter(torch.ones(2*dim_embd,2*dim_embd))
        self.vars_z_in.append(weight_med_z)
        #weight 11
        weight_enc_z=nn.Parameter(torch.ones(2*dim_embd,2*dim_embd))
        self.vars_z_in.append(weight_enc_z)
        ##################
        self.out_h_enc=nn.Linear(dim_embd,dim_embd,bias=False)
        self.out_x_enc=nn.Linear(dim_embd,dim_embd,bias=False)
        
        self.out_h_lab=nn.Linear(dim_embd,dim_embd,bias=False)
        self.out_x_lab=nn.Linear(dim_embd,dim_embd,bias=False)
        
        self.out_h_input=nn.Linear(dim_embd,dim_embd,bias=False)
        self.out_x_input=nn.Linear(dim_embd,dim_embd,bias=False)
        
        self.out_h_med=nn.Linear(dim_embd,dim_embd,bias=False)
        self.out_x_med=nn.Linear(dim_embd,dim_embd,bias=False)
        ##################
        
    def reset_params(self):
        if not self.nodecay and not self.basic:
            #enc_decay
            torch.nn.init.kaiming_normal_(self.W_decay_enc.weight)
            self.W_decay_enc.bias.data.zero_()
            #med_decay
            torch.nn.init.kaiming_normal_(self.W_decay_med.weight)
            self.W_decay_med.bias.data.zero_()
            #inputs_decay
            torch.nn.init.kaiming_normal_(self.W_decay_inputs.weight)
            self.W_decay_inputs.bias.data.zero_()
        for i in range(len(self.vars_r_in)):
            torch.nn.init.kaiming_normal_(self.vars_r_in[i])
        for i in range(len(self.vars_out)):
            torch.nn.init.kaiming_normal_(self.vars_out[i])
        for i in range(len(self.vars_z_in)):
            torch.nn.init.kaiming_normal_(self.vars_z_in[i])
        torch.nn.init.kaiming_normal_(self.out_h_enc.weight)
        torch.nn.init.kaiming_normal_(self.out_x_enc.weight)
        torch.nn.init.kaiming_normal_(self.out_h_lab.weight)
        torch.nn.init.kaiming_normal_(self.out_x_lab.weight)
        torch.nn.init.kaiming_normal_(self.out_h_input.weight)
        torch.nn.init.kaiming_normal_(self.out_x_input.weight)
        torch.nn.init.kaiming_normal_(self.out_h_med.weight)
        torch.nn.init.kaiming_normal_(self.out_x_med.weight)
        
    def forward(self, enc_med,enc_inputs,enc_lab,decay_mask_lower, time,lengths, h_e_r=None,h_m_r=None,h_l_r=None,h_i_r=None):
        h_e=h_e_r
        h_m=h_m_r
        h_l=h_l_r
        h_i=h_i_r
        batch_size=enc_med.shape[0]
        if h_e is None:
            h_e=torch.zeros(batch_size,enc_med.shape[1],self.dim_embd)
        if h_m is None:
            h_m=torch.zeros(batch_size,enc_med.shape[-1],self.dim_embd)
        if h_l is None:
            h_l=torch.zeros(batch_size,enc_lab.shape[-1],self.dim_embd)
        if h_i is None:
            h_i=torch.zeros(batch_size,enc_inputs.shape[-1],self.dim_embd)
        if self.vars_out[0].is_cuda and not h_e.is_cuda:
            h_e=h_e.cuda()
        if self.vars_out[0].is_cuda and not h_m.is_cuda:
            h_m=h_m.cuda()
        if self.vars_out[0].is_cuda and not h_l.is_cuda:
            h_l=h_l.cuda()
        if self.vars_out[0].is_cuda and not h_i.is_cuda:
            h_i=h_i.cuda()
        ###################
        #decay_mask_lower=lower_decay(time,max(lengths))
        ##decay_mask_upper=upper_decay(time,max(lengths))
        #if self.vars_out[0].is_cuda and not decay_mask_lower.is_cuda:
        #    decay_mask_lower=decay_mask_lower.cuda()
        decay_mask_lower = decay_mask_lower.unsqueeze(dim = -1)
        for loop in range(self.loop_num):
            if not self.nodecay and not self.basic:
                #enc decay
                h_e_temp=lower_(h_e,lengths)
                h_e_r_s=torch.tanh(self.W_decay_enc(h_e_temp))
                #decay1  wrong!
                #h_e_r_s_=lower_(h_e_r_s)*decay_mask
                #h_e_r_t=h_e-lower_(h_e_r_s)
                #decay2
                h_e_r_s_=h_e_r_s*decay_mask_lower
                h_e_r_t=h_e_temp-h_e_r_s
                ######
                h_e_r_adj=h_e_r_s_+h_e_r_t
                #med decay
                h_nm_temp=torch.matmul(lower_(enc_med,lengths),h_m)
                h_nm_r_s=torch.tanh(self.W_decay_med(h_nm_temp))
                h_nm_r_s_=h_nm_r_s*decay_mask_lower
                h_nm_r_t=h_nm_temp-h_nm_r_s
                h_nm_adj=h_nm_r_s_+h_nm_r_t
                #inputs decay
                #h_ni=torch.matmul(lower_(enc_inputs),h_i)
                h_ni_temp=torch.matmul(lower_(enc_inputs,lengths),h_i)
                h_ni_r_s=torch.tanh(self.W_decay_inputs(h_ni_temp))
                h_ni_r_s_=h_ni_r_s*decay_mask_lower
                h_ni_r_t=h_ni_temp-h_ni_r_s
                h_ni_adj=h_ni_r_s_+h_ni_r_t
            elif self.nodecay and not self.basic:
                #enc decay
                h_e_r_adj=lower_(h_e,lengths)
                #med decay
                h_nm_adj=torch.matmul(lower_(enc_med,lengths),h_m)
                #inputs decay
                h_ni_adj=torch.matmul(lower_(enc_inputs,lengths),h_i)
            ###################
            h_lab2enc=F.leaky_relu(torch.matmul(torch.matmul(enc_lab,h_l),self.vars_out[0]))
            h_inputs2enc=F.leaky_relu(torch.matmul(torch.matmul(enc_inputs,h_i),self.vars_out[1]))
            h_med2enc=F.leaky_relu(torch.matmul(torch.matmul(enc_med,h_m),self.vars_out[2]))
            h_enc2lab=F.leaky_relu(torch.matmul(torch.matmul(enc_lab.transpose(-1,-2),h_e),self.vars_out[3]))
            h_enc2med=F.leaky_relu(torch.matmul(torch.matmul(enc_med.transpose(-1,-2),h_e),self.vars_out[3]))
            h_enc2inputs=F.leaky_relu(torch.matmul(torch.matmul(enc_inputs.transpose(-1,-2),h_e),self.vars_out[3]))
            if not self.basic:
                h_enc2nmed=F.leaky_relu(torch.matmul(torch.matmul(upper_(enc_med).transpose(-1,-2),h_e),self.vars_out[3]))
                h_enc2ninputs=F.leaky_relu(torch.matmul(torch.matmul(upper_(enc_inputs).transpose(-1,-2),h_e),self.vars_out[3]))
            ###################
            if not self.basic:
                h_e_r_c=torch.cat([torch.unsqueeze(h_e_r_adj,dim=2),torch.unsqueeze(h_lab2enc,dim=2),\
                                   torch.unsqueeze(h_inputs2enc,dim=2),torch.unsqueeze(h_med2enc,dim=2),\
                                       torch.unsqueeze(h_nm_adj,dim=2),torch.unsqueeze(h_ni_adj,dim=2)],dim=2)
                h_l_r_c=torch.cat([torch.unsqueeze(h_enc2lab,dim=2)],dim=2)
                h_i_r_c=torch.cat([torch.unsqueeze(h_enc2inputs,dim=2),torch.unsqueeze(h_enc2ninputs,dim=2)],dim=2)
                h_m_r_c=torch.cat([torch.unsqueeze(h_enc2med,dim=2),torch.unsqueeze(h_enc2nmed,dim=2)],dim=2)
            else:
                h_e_r_c=torch.cat([torch.unsqueeze(h_lab2enc,dim=2),torch.unsqueeze(h_inputs2enc,dim=2),torch.unsqueeze(h_med2enc,dim=2)],dim=2)
                h_l_r_c=torch.cat([torch.unsqueeze(h_enc2lab,dim=2)],dim=2)
                h_i_r_c=torch.cat([torch.unsqueeze(h_enc2inputs,dim=2)],dim=2)
                h_m_r_c=torch.cat([torch.unsqueeze(h_enc2med,dim=2)],dim=2)
            ###################
            h_e_r_r=F.softmax(torch.matmul(torch.cat([torch.ones(h_e_r_c.shape).cuda().copy_(torch.unsqueeze(h_e,dim=2)),\
                                         h_e_r_c],dim=-1),self.vars_r_in[3]),dim=2)
            h_l_r_r=F.softmax(torch.matmul(torch.cat([torch.ones(h_l_r_c.shape).cuda().copy_(torch.unsqueeze(h_l,dim=2)),\
                                         h_l_r_c],dim=-1),self.vars_r_in[0]),dim=2)
            h_i_r_r=F.softmax(torch.matmul(torch.cat([torch.ones(h_i_r_c.shape).cuda().copy_(torch.unsqueeze(h_i,dim=2)),\
                                         h_i_r_c],dim=-1),self.vars_r_in[1]),dim=2)
            h_m_r_r=F.softmax(torch.matmul(torch.cat([torch.ones(h_m_r_c.shape).cuda().copy_(torch.unsqueeze(h_m,dim=2)),\
                                         h_m_r_c],dim=-1),self.vars_r_in[2]),dim=2)   
            ####################
            h_e_=torch.squeeze(torch.sum(h_e_r_r*h_e_r_c,dim=2),dim=2)
            h_l_=torch.squeeze(torch.sum(h_l_r_r*h_l_r_c,dim=2),dim=2)
            h_i_=torch.squeeze(torch.sum(h_i_r_r*h_i_r_c,dim=2),dim=2)
            h_m_=torch.squeeze(torch.sum(h_m_r_r*h_m_r_c,dim=2),dim=2)
            ####################
            h_e_r_outs=F.sigmoid(torch.matmul(torch.cat([h_e,h_e_],dim=-1),self.vars_z_in[3]))
            h_l_r_outs=F.sigmoid(torch.matmul(torch.cat([h_l,h_l_],dim=-1),self.vars_z_in[0]))
            h_i_r_outs=F.sigmoid(torch.matmul(torch.cat([h_i,h_i_],dim=-1),self.vars_z_in[1]))
            h_m_r_outs=F.sigmoid(torch.matmul(torch.cat([h_m,h_m_],dim=-1),self.vars_z_in[2]))
            ####################
            h_e_r_r,h_e_r_z=torch.chunk(h_e_r_outs,2,-1)
            h_l_r_r,h_l_r_z=torch.chunk(h_l_r_outs,2,-1)
            h_i_r_r,h_i_r_z=torch.chunk(h_i_r_outs,2,-1)
            h_m_r_r,h_m_r_z=torch.chunk(h_m_r_outs,2,-1)
            ####################
            h_e_=torch.tanh(self.out_h_enc(h_e_r_r*h_e)+self.out_x_enc(h_e_))
            h_l_=torch.tanh(self.out_h_lab(h_l_r_r*h_l)+self.out_x_lab(h_l_))
            h_i_=torch.tanh(self.out_h_input(h_i_r_r*h_i)+self.out_x_input(h_i_))
            h_m_=torch.tanh(self.out_h_med(h_m_r_r*h_m)+self.out_x_med(h_m_))
            ####################
            #h=(1-z)*h_adj+z*h_
            h_e=((1.0-h_e_r_z)*h_e+h_e_r_z*torch.tanh(h_e_))
            h_l=((1.0-h_l_r_z)*h_l+h_l_r_z*torch.tanh(h_l_))
            h_m=((1.0-h_m_r_z)*h_m+h_m_r_z*torch.tanh(h_m_))
            h_i=((1.0-h_i_r_z)*h_i+h_i_r_z*torch.tanh(h_i_))
            '''
            if self.vars_out[0].is_cuda and not h_e.is_cuda:
                h_e=h_e.cuda()
            if self.vars_out[0].is_cuda and not h_l.is_cuda:
                h_l=h_l.cuda()
            if self.vars_out[0].is_cuda and not h_m.is_cuda:
                h_m=h_m.cuda()
            if self.vars_out[0].is_cuda and not h_i.is_cuda:
                h_i=h_i.cuda()
            '''
        return h_e, h_m, h_l, h_i
