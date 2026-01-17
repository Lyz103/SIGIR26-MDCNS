# -*- coding: utf-8 -*-
# @Time    : 2020/4/25 22:59
# @Author  :  

import os
import numpy as np
import random
import torch
import argparse

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import SRDataset, DNSDataset
from trainers import FinetuneTrainer
from models import SASRec, Mamba4Rec, GRU4Rec, TTT4Rec, Narm, Linrec, LightSANs, FMLPRecModel
from utils import *
import time
time_stmp = time.time()


'''实现其他的负采样基线方法'''
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from collections import deque, defaultdict
import random
# 1. SRNS

class HisScore:
    def __init__(self,cap):
        self.cap = cap
        # self.size = 0
        self.his_scores = defaultdict(deque)

    def compute_std(self,data_idx,mu_idx):
        # mu_idx [batch,s1]
        device = mu_idx.device
        mu_his_scores = []
        data_idx = data_idx.cpu().tolist() # [batch_size]
        # 提取当前的candi_his_score，并拼成tensor
        batch_candi_his = []
        for idx in data_idx:
            # batch_candi_his.append(list(self.his_scores[idx]))
            data_his = list(self.his_scores[idx]) # list中是tensor [candi_num]
            data_his = torch.stack(data_his,dim=0) #[n,candi_num]
            batch_candi_his.append(data_his)
        batch_candi_his = torch.stack(batch_candi_his,dim=0).to(device) # [batch,n,candi_num]

        # 从mu_idx gather到his score [batch,n,s1]
        batch_size = mu_idx.shape[0]
        s1 = mu_idx.shape[-1]
        n = batch_candi_his.shape[1]
        mu_expand = mu_idx.unsqueeze(1).expand(batch_size,n,s1)  # [batch,n,s1]
        select_his_scores = batch_candi_his.gather(-1,mu_expand) # # [batch,n,s1]

        # 沿着n求std #[BATCH,S1]
        his_std = torch.std(select_his_scores,dim=1) #

        return his_std

    def update_score(self,data_idx,batch_new_scores):
        # batch_new_scores [batch,candi_num]
        data_idx = data_idx.cpu().tolist()
        for i,idx in enumerate(data_idx):
            temp = self.his_scores[idx] # deque
            if len(temp) >= self.cap:
                temp.popleft()
            temp.append(batch_new_scores[i].cpu())



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--output_dir', default='output/', type=str)
    parser.add_argument('--data_name', default='Beauty', type=str)
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--ckp', default=0, type=int, help="pretrain epochs 10, 20, 30...")
    parser.add_argument('--num_split', type=int, help='number of split', default=6)


    # model args
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument('--hidden_act', default="gelu", type=str) # gelu relu
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=50, type=int)
    parser.add_argument('--backbone', type=str, help="SASRec,Mamba4Rec,GRU4Rec...")
    # Hyperparameters for Attention block
    parser.add_argument('--num_attention_heads', default=2, type=int) 
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    # Hyperparameters for Mamba block
    parser.add_argument('--d_state', type=int, help='Dimension of state', default=32)
    parser.add_argument('--d_conv', type=int, help='Dimension of convolution', default=4)
    parser.add_argument('--expand', type=int, help='Expansion factor', default=2)
    # Hyperparameters for GRU block
    parser.add_argument('--embedding_size', type=int, help='Size of the embedding layer', default=64)
    # Hyperparameters for TTT block
    parser.add_argument('--num_TTT_heads', type=int, help='Number of TTT heads', default=4)
    parser.add_argument('--mini_batch_size', type=int, help='Size of mini batches', default=16)
    parser.add_argument('--rope_theta', type=float, help='Theta value for relative position encoding', default=10000)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=1024, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)


    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")


    # Neg sample
    parser.add_argument("--N", type=int, default=1000, help="sample_size")
    parser.add_argument("--M", type=int, default=10, help="pool_size")
    parser.add_argument("--neg_sampler", type=str, default="DNS", help="neg_sampler")
    parser.add_argument("--loss_type", type=str, default="BCE", help="BCE,BPR,SFM")
    parser.add_argument("--CL_type", type=str, default="Radical", help="Radical, Gentle")
    parser.add_argument("--start_epoch", default=30, type=int)
    parser.add_argument("--K", default=0.05, type=int)

    parser.add_argument('--warm', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=5)
    parser.add_argument('--s', type=int, default=20)
    parser.add_argument('--cap', type=int, default=5)



    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)

    other_dict = {
    'warm':args.warm,
    'alpha':args.alpha,
    's1':args.s,
    's2':args.s
    }
    args.kwargs = other_dict

    args.cuda_condition = True

    args.data_file = args.data_dir + args.data_name

    item_size = 0
    args.data_file = args.data_dir + args.data_name
    train_data, max_item, _ = get_user_seqs(args.data_file + "_train.txt")
    item_size = max(item_size, max_item)
    valid_data, max_item, _ = get_user_seqs(args.data_file + "_val.txt")
    item_size = max(item_size, max_item)
    test_data, max_item, _ = get_user_seqs(args.data_file + "_test.txt")
    item_size = max(item_size, max_item)
    args.item_size = item_size + 2
    valid_matrix = generate_rating_matrix(valid_data, args.item_size)
    test_matrix = generate_rating_matrix(test_data, args.item_size)

    # save model args
    args_str = f'{args.backbone}-{args.data_name}-{args.ckp}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    print(str(args))
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    # set item score in train set to `0` in validation
    args.train_matrix = valid_matrix

    # save model
    checkpoint = args_str + str(time_stmp) + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    neg_sampler_dict = {
        'Uniform': SRDataset,
        "DNS": DNSDataset
    }

    # ! 定义mu和his_scores
    his_scores = HisScore(args.cap)
    total_data = len(train_data)
    Mu_idx = []  # All possible items or non-fn items
    for i in tqdm(range(total_data)):
        Mu_idx_tmp = random.sample(list(range(args.N)), args.s)
        Mu_idx.append(Mu_idx_tmp)
    # mu = torch.LongTensor(Mu_idx).to(accelerator.device)
    # mu = torch.LongTensor(Mu_idx)
    mu = torch.tensor(Mu_idx, dtype=torch.int32)
    args.his_scores = his_scores
    args.mu = mu


    print("neg_sampler: ", args.neg_sampler)
    train_dataset = DNSDataset(args, train_data ,args.N)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=8)

    eval_dataset = neg_sampler_dict[args.neg_sampler](args, valid_data)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    test_dataset = neg_sampler_dict[args.neg_sampler](args, test_data)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    # 创建模型名称到类的映射
    model_dict = {
        'SASRec': SASRec,
        'Mamba4Rec': Mamba4Rec,
        'GRU4Rec': GRU4Rec,
        'TTT4Rec': TTT4Rec,
        'Narm': Narm,
        'Linrec': Linrec,
        'LightSANs': LightSANs,
        'FMLPRecModel': FMLPRecModel
    }
    if args.backbone in model_dict:
        model = model_dict[args.backbone](args=args)
    else:
        raise ValueError("Unsupported backbone model name: {}".format(args.backbone))

    trainer = FinetuneTrainer(model, train_dataloader, eval_dataloader,
                              test_dataloader, args)


    if args.do_eval:
        trainer.load(args.checkpoint_path)
        print(f'Load model from {args.checkpoint_path} for test!')
        trainer.args.train_matrix = test_matrix
        scores, result_info = trainer.test(0, full_sort=True)

    else:
        Epoch = 0
        early_stopping = EarlyStopping(args.checkpoint_path, patience=50, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            # evaluate on NDCG@20
            scores, _ = trainer.valid(epoch, full_sort=True)
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                Epoch = epoch
                break
            Epoch = epoch
            
        trainer.args.train_matrix = test_matrix
        print('---------------Change to test_rating_matrix!-------------------')
        # load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0, full_sort=True)

    print(args_str)
    print(result_info)
    to_excel(result_info, args, args.start_epoch, Epoch, training_time=0, inference_time=0)
    with open(args.log_file, 'a') as f:
        f.write(args_str + '\n')
        f.write(result_info + '\n')
main()