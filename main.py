#main.py
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time as timem
import numpy as np
from model import Trainer
from param_parser import parameter_parser


def main():
    args = parameter_parser()
    print('loading trainer ......')
    trainer = Trainer(args)
    print('finish!')
    times_list = []
    start = timem.time()
    try:
        os.mkdir(trainer.args.path+'net_params//'+trainer.DATE+'//')
    except:
        print('Folder already exists!')
    trainer.printf('\n\n\nNow running '+trainer.DATE+' ......')
    accs, baccs, b_accs, bacc_s, b_acc_s, rocs, brocs, b_rocs, broc_s, b_roc_s, praucs, bpraucs, b_praucs, bprauc_s, b_prauc_s = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    for f in range(args.folds_num):
        time = trainer.fit(f)
        times_list.append(time)
        acc, bacc, b_acc, bacc_, b_acc_, roc, broc, b_roc, broc_, b_roc_, prauc, bprauc, b_prauc, bprauc_, b_prauc_ = trainer.test()
        accs.append(acc)
        baccs.append(bacc)
        b_accs.append(b_acc)
        bacc_s.append(bacc_)
        b_acc_s.append(b_acc_)
        rocs.append(roc)
        brocs.append(broc)
        b_rocs.append(b_roc)
        broc_s.append(broc_)
        b_roc_s.append(b_roc_)
        praucs.append(prauc)
        bpraucs.append(bprauc)
        b_praucs.append(b_prauc)
        bprauc_s.append(bprauc_)
        b_prauc_s.append(b_prauc_)
    trainer.printf('========================================================')
    trainer.printf('ACCs: '+str(accs)+'\t'+str(np.mean(accs)))
    trainer.printf('Best ACCs: '+str(baccs)+'\t'+str(np.mean(baccs)))
    trainer.printf('Best_ACCs: '+str(b_accs)+'\t'+str(np.mean(b_accs)))
    trainer.printf('Best ACC_s: '+str(bacc_s)+'\t'+str(np.mean(bacc_s)))
    trainer.printf('Best_ACC_s: '+str(b_acc_s)+'\t'+str(np.mean(b_acc_s)))
    
    trainer.printf('AUC_ROCs: '+str(rocs)+'\t'+str(np.mean(rocs)))
    trainer.printf('Best AUC_ROCs: '+str(brocs)+'\t'+str(np.mean(brocs)))
    trainer.printf('Best_AUC_ROCs: '+str(b_rocs)+'\t'+str(np.mean(b_rocs)))
    trainer.printf('Best AUC_ROC_s: '+str(broc_s)+'\t'+str(np.mean(broc_s)))
    trainer.printf('Best_AUC_ROC_s: '+str(b_roc_s)+'\t'+str(np.mean(b_roc_s)))
    
    trainer.printf('AUC_PRs: '+str(praucs)+'\t'+str(np.mean(praucs)))
    trainer.printf('Best AUC_PRs: '+str(bpraucs)+'\t'+str(np.mean(bpraucs)))
    trainer.printf('Best_AUC_PRs: '+str(b_praucs)+'\t'+str(np.mean(b_praucs)))
    trainer.printf('Best AUC_PR_s: '+str(bprauc_s)+'\t'+str(np.mean(bprauc_s)))
    trainer.printf('Best_AUC_PR_s: '+str(b_prauc_s)+'\t'+str(np.mean(b_prauc_s)))
    
    trainer.printf('Times: '+str(times_list)+'\t'+str(np.mean(times_list)))
    trainer.printf('=====================FINISH===========================')
    end=timem.time()
    trainer.printf(str(int((end-start)/3600))+'h '+str(int(((end-start)%3600)/60))+'min '+str(int(((end-start)%3600)%60))+'s')

if __name__ == '__main__':
    main()
