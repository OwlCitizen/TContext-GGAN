#param_parser.py
import argparse

def parameter_parser():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--task", type = str, default='mortality48_hour_iv_hf')
    
    parser.add_argument("--wp", type = int, default=7)
    
    parser.add_argument("--folds_num", type = int, default=5)
    
    parser.add_argument("--BATCH_SIZE", type = int, default=128)
    
    parser.add_argument("--LR", type = float, default=1e-4)
    
    parser.add_argument("--MAX_EPOCH", type = int, default=1000)
    
    parser.add_argument("--path", type = str, default='.//')
    
    parser.add_argument("--patience", type = int, default=10)
    
    parser.add_argument("--val_epoch", type = int, default=1)
    
    parser.add_argument("--early_stop", type = str, default='acc')

    parser.add_argument("--dim_lab", type = int, default=32)
    
    parser.add_argument("--dim_inputs", type = int, default=16)
    
    parser.add_argument("--dim_med", type = int, default=16)
    
    parser.add_argument("--dim_embd1", type = int, default=8)
    
    parser.add_argument("--num_layer", type = int, default=2)
    
    parser.add_argument("--loop_num", type = int, default=1)
    
    parser.add_argument("--nodecay", type = bool, default=False)
    
    parser.add_argument("--basic", type = bool, default=False)
    
    parser.add_argument("--qkv_diff", type = bool, default=False)
    
    parser.add_argument("--dp_rate", type = float, default=0.5)
    
    parser.add_argument("--pass_v", type = bool, default=True)
     
    parser.add_argument("--info_name", type = str, default='folds_info')

    return parser.parse_args()
