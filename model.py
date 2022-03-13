#model.py
import os
import torch
import time as timem
from torch import nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torch.utils.data as Data
from torch.optim.lr_scheduler import MultiStepLR, StepLR, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from layers import TGGANN as GCNLayer
from layers import SelfAttentionLayer
from cols import cols, lab, medi, inputs
from dataset import HFDataset
from utils import visit_collate_fn_
from tensorboardX import SummaryWriter


class TContextGGANN(nn.Module):
    def __init__(self, dim_lab, dim_inputs, dim_med, dim_embd1, num_layer=2, loop_num=1, \
                 nodecay=False, basic=False, qkv_diff=True, dp_rate=0.5, pass_v=True):
        super(TContextGGANN, self).__init__()
        self.dim_lab=dim_lab
        self.dim_med=dim_med
        self.dim_inputs=dim_inputs
        self.dim_embd=dim_lab+dim_inputs+dim_med
        self.num_layer = num_layer
        
        self.embd_lab=nn.Linear(len(lab),dim_lab)
        self.embd_inputs=nn.Linear(len(inputs),dim_inputs)
        self.embd_med=nn.Linear(len(medi),dim_med)
        
        self.gats=nn.ModuleList()
        for _ in range(self.num_layer):
            self.gats.append(GCNLayer(dim_embd=self.dim_embd, loop_num=loop_num, nodecay=nodecay, basic=basic))
            
        self.saLayer=SelfAttentionLayer(dim_embd=self.dim_embd, dp_rate=dp_rate, qkv_diff=qkv_diff, pass_v = pass_v, out=True)
        
        self.beta_fc=nn.Linear(self.dim_embd,dim_embd1)
        self.output=nn.Linear(in_features=dim_embd1, out_features=2)
    
    def reset_params(self):
        torch.nn.init.kaiming_normal_(self.embd_lab.weight)
        self.embd_lab.bias.data.zero_()
        torch.nn.init.kaiming_normal_(self.embd_inputs.weight)
        self.embd_inputs.bias.data.zero_()
        torch.nn.init.kaiming_normal_(self.embd_med.weight)
        self.embd_med.bias.data.zero_()
        for l in range(self.num_layer):
            self.gats[l].reset_params()
        self.saLayer.reset_params()
        torch.nn.init.kaiming_normal_(self.beta_fc.weight)
        self.beta_fc.bias.data.zero_()
        torch.nn.init.kaiming_normal_(self.output.weight)
        self.output.bias.data.zero_()
    
    def forward(self, batch):
        (data, decay, time, label, lab_mask, length, pid)=batch
        batch_size=data.shape[0]
        lab_batch=data[:,:,:len(lab)]
        input_batch=data[:,:,len(lab):(len(lab)+len(inputs))]
        med_batch=data[:,:,(len(lab)+len(inputs)):]
        labels=label.to(torch.int64).long()
        labels=labels.cuda()
        decay = decay.cuda()
        time_batch=time
        lab_batch=torch.tensor(lab_batch, dtype=torch.float32).cuda()
        input_batch=torch.tensor(input_batch, dtype=torch.float32).cuda()
        med_batch=torch.tensor(med_batch, dtype=torch.float32).cuda()
        lab_mask=torch.tensor(lab_mask, dtype=torch.float32).cuda()
        
        embd_lab=self.embd_lab(lab_mask)
        embd_inputs=self.embd_inputs((input_batch!=0).to(torch.float32).cuda())
        embd_med=self.embd_med(med_batch)
        embd=torch.cat([embd_lab,embd_inputs,embd_med],dim=-1)
        
        ih_e=embd
        ih_l=torch.zeros([batch_size,len(lab),self.dim_embd]).copy_(torch.cat([self.embd_lab(torch.eye(len(lab)).cuda()),torch.zeros(len(lab),self.dim_med+self.dim_inputs).cuda()],dim=1)).to(torch.float32)
        ih_m=torch.zeros([batch_size,len(medi),self.dim_embd]).copy_(torch.cat([torch.zeros(len(medi),self.dim_lab+self.dim_inputs).cuda(),self.embd_med(torch.eye(len(medi)).cuda())],dim=1)).to(torch.float32)
        ih_i=torch.zeros([batch_size,len(inputs),self.dim_embd]).copy_(torch.cat([torch.zeros(len(inputs),self.dim_lab).cuda(),self.embd_inputs(torch.eye(len(inputs)).cuda()),torch.zeros(len(inputs),self.dim_med).cuda()],dim=1)).to(torch.float32)
        
        h_e, h_m, h_l, h_i = ih_e, ih_m, ih_l, ih_i
        for l in range(self.num_layer-1):
            h_e, h_m, h_l, h_i=self.gats[l](med_batch,input_batch,lab_batch,decay,time_batch,length,h_e,h_m,h_l,h_i)
            h_e, h_m, h_l, h_i=F.leaky_relu(h_e),F.leaky_relu(h_m),F.leaky_relu(h_l),F.leaky_relu(h_i)
        h_e, h_m, h_l, h_i=self.gats[self.num_layer-1](med_batch,input_batch,lab_batch,decay,time_batch,length,h_e,h_m,h_l,h_i)
        
        h_e=self.saLayer(h_e,h_l,h_i,h_m,length)
        
        beta=torch.tanh(self.beta_fc(h_e))
        
        h=[]
        for b, blength in enumerate(length):
            h_=beta[b:b+1,blength-1,:]
            h.append(h_)
        h = torch.cat(h, dim=0)
        
        logit = self.output(h)
        logit = F.softmax(logit)
        return logit, labels

class Trainer(nn.Module):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.net = TContextGGANN(args.dim_lab, args.dim_inputs, args.dim_med, \
                                 args.dim_embd1, args.num_layer, args.loop_num, \
                                     args.nodecay, args.basic, args.qkv_diff, \
                                         args.dp_rate, args.pass_v).cuda()
        self.dataset = HFDataset(args.task, args.info_name, args.wp)
        self.loss_func = ''
        self.scheduler = ''
        self.optimizer = ''
        self.writer = ''
        self.args = args
        self.DATE = 'TContext_'+timem.strftime("%Y%m%d", timem.localtime())[2:]+\
            '_bs'+str(args.BATCH_SIZE)+'_lr'+str(args.LR)+'_epoch'+str(args.MAX_EPOCH)+\
                '_weight_'+str(args.wp)+('_qkvdiff' if self.args.qkv_diff else '')+\
                    ('_basic' if self.args.basic else '')+('_nodecay' if self.args.nodecay else '')+\
                        '_'+args.early_stop+'Earlystop'
    
    def printf(self,text):
        file = open(".//results//"+self.DATE+".txt", "a+")
        file.write(text+'\n')
        file.close()
        print(text)
    
    def printf_(self,text):
        file = open(".//results//"+self.DATE+"_log.txt", "a+")
        file.write(text+'\n')
        file.close()
        print(text) 
    
    def fit(self, f=0):
        start = timem.time()
        #times_list = []
        #for f in self.args.folds_num:
        try:
            os.mkdir(self.args.path+'net_params//'+self.DATE+'//fold'+str(f)+'//')
        except:
            print('Folder already exists!')
        self.printf('Now running fold '+str(f)+' ...')
        self.net.reset_params()
        self.dataset.initfold(f)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.LR)   # optimize all parameters
        self.loss_func = FocalLoss(weight=self.dataset.train_w.cuda())#weight=dataset.train_w.cuda())#.cuda()   # the target label is not one-hotted
        self.scheduler = MultiStepLR(self.optimizer, milestones=[20,40,60,80,100], gamma=0.5)
        self.writer = SummaryWriter(self.args.path+'tensorboard//'+self.DATE+'//fold'+str(f))
        
        print('Start trainning ...')
        fstart=timem.time()
        patience=self.args.patience
        best_vacc=0.0
        best_vauc=0.0
        best_vprauc=0.0
        best_vloss=1000.0
        self.best_epoch_acc=0
        self.best_epoch_roc=0
        self.best_epoch_prauc=0
        self.best_epoch_loss=0
        self.best_params_acc=[]
        self.best_params_roc=[]
        self.best_params_prauc=[]
        self.best_params_loss=[]
        
        for epoch in range(self.args.MAX_EPOCH):
            self.net.train()
            self.dataset.initmode('train')
            train_loader = Data.DataLoader(
                dataset=self.dataset,      # torch TensorDataset format
                batch_size=self.args.BATCH_SIZE,      # mini batch size
                shuffle=True,               # 要不要打乱数据 (打乱比较好)
                #num_workers=4,              # 多线程来读数据
                collate_fn=visit_collate_fn_,
            )
            total=int(len(self.dataset)/self.args.BATCH_SIZE)+1
            count=0
            saccs=[]
            slosses=[]
            scount=0
            
            preds=[]
            y_true=[]
            losses=[]
            for step, batch in enumerate(train_loader):
                scount+=1
                sstart=timem.time()
                print('\nNow running fold '+str(f)+' epoch '+str(epoch)+'\tstep '+str(step)+'/'+str(total)+'\t'+str(int(step*100/total))+'%')
                outs,labels=self.net(batch)
                loss=self.loss_func(outs,labels)
                self.optimizer.zero_grad()           # clear gradients for this training step
                loss.backward()                 # backpropagation, compute gradients
                self.optimizer.step()                # apply gradients
                send=timem.time()
                preds.append(outs.clone().detach_().cpu())
                y_true.append(labels.clone().detach_().cpu())
                
                pred_y = torch.max(outs.clone().detach().cpu(), 1)[1].data.numpy()
                acc=sum(pred_y==labels.clone().detach().cpu().data.numpy())*100.0/len(labels)
                saccs.append(acc)
                slosses.append(loss.clone().detach_().cpu().item())
                losses.append(loss.clone().detach_().cpu().item())
                
                print('Loss: '+str(loss.clone().detach().cpu().item())+'|ACC: '+str(acc)+'|Patience: '+str(patience)+'|STEP TAKES:'+str(int(((send-sstart)%3600)%60))+'s'\
                          +'|FOLD HAS RUN:'+str(int((send-fstart)/3600))+'h '+str(int(((send-fstart)%3600)/60))+'min '+str(int(((send-fstart)%3600)%60))+'s'\
                              +'|HAS RUN:'+str(int((send-start)/3600))+'h '+str(int(((send-start)%3600)/60))+'min '+str(int(((send-start)%3600)%60))+'s')
                
                if scount==5:
                    self.writer.add_scalar('00.Loss', np.mean(slosses),(epoch*total+step))
                    self.writer.add_scalar('00.Acc', np.mean(saccs),(epoch*total+step))
                    slosses=[]
                    saccs=[]
                    scount=0
            
            outs = torch.cat(preds, dim = 0)
            labels = torch.cat(y_true, dim = 0)
            preds = outs.argmax(dim = 1)
            eacc = sum(preds == labels)/len(labels)
            eauc = roc_auc_score(labels.numpy(), F.softmax(outs)[:,1].numpy())
            eloss = sum(losses)/len(losses)
            
            params=list(self.net.named_parameters())
            for p in range(len(params)):
                num=p+2
                if num<10:
                    num='0'+str(num)
                else:
                    num=str(num)
                self.writer.add_scalar(num+'.Parameter '+str(p)+':'+str(params[p][0]), torch.mean(params[p][1]),(epoch*total+step))
                if params[p][1].grad is None:
                    self.printf_('Warning:fold '+str(f)+' epoch '+str(epoch)+' step '+str(step)+' params '+str(p)+'\t'+str(params[p].shape)+'\t grad is None!')
                else:
                    self.writer.add_scalar(num+'.Parameter '+str(p)+' grad:'+str(params[p][0]), torch.mean(params[p][1].grad),(epoch*total+step))
                    
            if(eloss<1e-10):
                count+=1
                        
            if epoch!=0 and epoch%self.args.val_epoch==0:
                self.dataset.initmode('val')
                val_loader = Data.DataLoader(
                    dataset=self.dataset,      # torch TensorDataset format
                    batch_size=1,      # mini batch size
                    shuffle=True,               # 要不要打乱数据 (打乱比较好)
                    #num_workers=4,              # 多线程来读数据
                    collate_fn=visit_collate_fn_,
                )
                total=int(len(self.dataset)/1)+1
                vacc, vroc, vprauc, vloss = self.evaluate(val_loader, epoch)
                vloss = vloss.clone().detach().cpu().item()
                
                self.writer.add_scalar('01.Acc Val', vacc,epoch)
                self.writer.add_scalar('01.Loss Val', vloss, epoch)
                self.writer.add_scalar('01.ROC Val', vroc,epoch)
                self.writer.add_scalar('01.PRAUC Val', vprauc,epoch)
                self.writer.add_scalar('01.Loss Train', eloss,epoch)
                self.writer.add_scalar('01.Acc Train', eacc,epoch)
                self.writer.add_scalar('01.AUC Train', eauc,epoch)
                
                if (vacc > best_vacc and patience > 0) or epoch==0:
                    self.best_params_acc=self.net.state_dict()
                    best_vacc=vacc
                    if self.args.early_stop == 'acc':
                        patience=self.args.patience
                    self.best_epoch_acc=epoch
                elif self.args.early_stop == 'acc':
                    patience -= 1
                
                if (vroc > best_vauc and patience > 0) or epoch==0:
                    self.best_params_roc=self.net.state_dict()
                    best_vauc=vroc
                    if self.args.early_stop == 'auc':
                        patience=self.args.patience
                    self.best_epoch_roc=epoch
                elif self.args.early_stop == 'auc':
                    patience -= 1
                
                if (vprauc > best_vprauc and patience > 0) or epoch==0:
                    self.best_params_prauc=self.net.state_dict()
                    best_vprauc=vprauc
                    if self.args.early_stop == 'prauc':
                        patience=self.args.patience
                    self.best_epoch_prauc=epoch
                elif self.args.early_stop == 'prauc':
                    patience -= 1
                
                if (vloss < best_vloss and patience > 0) or epoch==0:
                    self.best_params_loss=self.net.state_dict()
                    best_vloss=vloss
                    if self.args.early_stop == 'loss':
                        patience=self.args.patience
                    self.best_epoch_loss=epoch
                elif self.args.early_stop == 'loss':
                    patience -= 1
                
                if(count>=5):
                    print('Epoch '+str(epoch)+' loss lost!')
                    break
                if patience<=0:
                    print('Patience out! Over Fitting!')
                    break
            
            print('Saving ...')
            torch.save(self.net.state_dict(), self.args.path+'net_params//'+self.DATE+'//fold'+str(f)+'//net_'+self.DATE+'_params.pkl')
            torch.save(self.best_params_acc, self.args.path+'net_params//'+self.DATE+'//fold'+str(f)+'//net_'+self.DATE+'_params(best_acc).pkl')
            torch.save(self.best_params_roc, self.args.path+'net_params//'+self.DATE+'//fold'+str(f)+'//net_'+self.DATE+'_params(best_auc).pkl')
            torch.save(self.best_params_loss, self.args.path+'net_params//'+self.DATE+'//fold'+str(f)+'//net_'+self.DATE+'_params(best_loss).pkl')
            torch.save(self.best_params_prauc, self.args.path+'net_params//'+self.DATE+'//fold'+str(f)+'//net_'+self.DATE+'_params(best_prauc).pkl')
            print('Save success!')    
            
            self.scheduler.step()
            #self.scheduler.step(loss)
        print('Trainning finish !')
        fend = timem.time()
        #times_list.append(fend-fstart)
        return fend-fstart
    
    def test(self):#, best_params, best_params_):
        #accs, baccs, b_accs, rocs, brocs, b_rocs, praucs, bpraucs, b_praucs = [], [], [], [], [], [], [], [], []
        with torch.no_grad():
            self.net.eval()
            self.dataset.initmode('test')
            test_loader = Data.DataLoader(
                dataset=self.dataset,      # torch TensorDataset format
                batch_size=1,      # mini batch size
                shuffle=True,               # 要不要打乱数据 (打乱比较好)
                #num_workers=4,              # 多线程来读数据
                collate_fn=visit_collate_fn_,
            )
            acc, roc, prauc, _ = self.evaluate(test_loader, 'final params')
            #accs.append(acc)
            #rocs.append(roc)
            #praucs.append(prauc)
            
            self.printf('=========================RESULT=========================')
            self.printf('ACC:'+str(acc)+'|AUC_ROC:'+str(roc)+'|AUC_PR:'+str(prauc))
            self.printf('========================================================')
            
            if self.best_epoch_acc!=0:
                self.net.load_state_dict(self.best_params_acc)
                self.net.eval()
                self.dataset.initmode('test')
                test_loader = Data.DataLoader(
                    dataset=self.dataset,      # torch TensorDataset format
                    batch_size=1,      # mini batch size
                    shuffle=True,               # 要不要打乱数据 (打乱比较好)
                    #num_workers=4,              # 多线程来读数据
                    collate_fn=visit_collate_fn_,
                )
                bacc, broc, bprauc, _ = self.evaluate(test_loader, 'best acc params')
                #baccs.append(bacc)
                #brocs.append(broc)
                #bpraucs.append(bprauc)
                
                self.printf('=========================RESULT (Best Acc '+str(self.best_epoch_acc)+')=========================')
                self.printf('ACC:'+str(bacc)+'|AUC_ROC:'+str(broc)+'|AUC_PR:'+str(bprauc))
                self.printf('===================================================================')
            
            if self.best_epoch_roc!=0:
                self.net.load_state_dict(self.best_params_roc)
                self.net.eval()
                self.dataset.initmode('test')
                test_loader = Data.DataLoader(
                    dataset=self.dataset,      # torch TensorDataset format
                    batch_size=1,      # mini batch size
                    shuffle=True,               # 要不要打乱数据 (打乱比较好)
                    #num_workers=4,              # 多线程来读数据
                    collate_fn=visit_collate_fn_,
                )
                b_acc, b_roc, b_prauc, _ = self.evaluate(test_loader, 'best auc params')
                #b_accs.append(bacc)
                #b_rocs.append(broc)
                #b_praucs.append(bprauc)
                
                self.printf('=========================RESULT (Best AUC '+str(self.best_epoch_roc)+')=========================')
                self.printf('ACC:'+str(b_acc)+'|AUC_ROC:'+str(b_roc)+'|AUC_PR:'+str(b_prauc))
                self.printf('===================================================================')
                
            if self.best_epoch_prauc!=0:
                self.net.load_state_dict(self.best_params_prauc)
                self.net.eval()
                self.dataset.initmode('test')
                test_loader = Data.DataLoader(
                    dataset=self.dataset,      # torch TensorDataset format
                    batch_size=1,      # mini batch size
                    shuffle=True,               # 要不要打乱数据 (打乱比较好)
                    #num_workers=4,              # 多线程来读数据
                    collate_fn=visit_collate_fn_,
                )
                bacc_, broc_, bprauc_, _ = self.evaluate(test_loader, 'best prauc params')
                #baccs.append(bacc)
                #brocs.append(broc)
                #bpraucs.append(bprauc)
                
                self.printf('=========================RESULT (Best PRAUC '+str(self.best_epoch_prauc)+')=========================')
                self.printf('ACC:'+str(bacc_)+'|AUC_ROC:'+str(broc_)+'|AUC_PR:'+str(bprauc_))
                self.printf('===================================================================')
            
            if self.best_epoch_loss!=0:
                self.net.load_state_dict(self.best_params_loss)
                self.net.eval()
                self.dataset.initmode('test')
                test_loader = Data.DataLoader(
                    dataset=self.dataset,      # torch TensorDataset format
                    batch_size=1,      # mini batch size
                    shuffle=True,               # 要不要打乱数据 (打乱比较好)
                    #num_workers=4,              # 多线程来读数据
                    collate_fn=visit_collate_fn_,
                )
                b_acc_, b_roc_, b_prauc_, _ = self.evaluate(test_loader, 'best loss params')
                #b_accs.append(bacc)
                #b_rocs.append(broc)
                #b_praucs.append(bprauc)
                
                self.printf('=========================RESULT (Best Loss '+str(self.best_epoch_loss)+')=========================')
                self.printf('ACC:'+str(b_acc_)+'|AUC_ROC:'+str(b_roc_)+'|AUC_PR:'+str(b_prauc_))
                self.printf('===================================================================')
            
        return acc, bacc, b_acc, bacc_, b_acc_, roc, broc, b_roc, broc_, b_roc_, prauc, bprauc, b_prauc, bprauc_, b_prauc_
                
    def evaluate(self, loader, epoch):
        self.net.eval()
        outs = []
        labels = []
        loss_func = nn.CrossEntropyLoss()
        with torch.no_grad():
            pbar = tqdm(enumerate(loader), desc = 'evaluating in epoch '+str(epoch))
            for step, batch in pbar:
                out, label = self.net(batch)
                outs.append(out.view(-1,2))
                labels.append(label.view(-1,1))
            outs = torch.cat(outs, dim = 0).clone().detach_().cpu()
            labels = torch.cat(labels, dim=0).clone().detach_().cpu().squeeze()
            preds = outs.argmax(dim = -1).numpy()
            
            loss = loss_func(outs, labels)
            acc = sum(preds == labels.numpy())*1.0/len(labels)
            #print(acc.shape)
            roc = roc_auc_score(labels.numpy(), outs[:,1].numpy())
            precision, recall, threshold = precision_recall_curve(labels.numpy(), preds)
            prauc = auc(recall, precision)
        
        return acc.squeeze(), roc, prauc, loss
    
    def save(self, path, params_dict=None):
        params_dict = self.net_state_dict() if params_dict is None else params_dict
        torch.save(params_dict, path)
        


class FocalLoss(nn.Module):
    def __init__(self,weight=None,gamma=2):
        super(FocalLoss, self).__init__()
        self.weight=weight
        self.gamma=gamma
    def forward(self, pred_y, targets):
        CE_loss=F.cross_entropy(pred_y, targets, weight=self.weight)
        mask=targets.float()*(pred_y[:,0]**self.gamma)+(1-targets.float())*(pred_y[:,1]**self.gamma)
        return torch.mean(mask*CE_loss)
