# -*- coding: utf-8 -*-
# @Time        : 2020/4/25 22:59
# @Author      : 
# (Original code provided by user)
#
# Modifications by Gemini to implement DWS-HNS framework.

import os
import numpy as np
import random
import torch
import argparse

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import SRDataset, DNSDataset
# Make sure you have MDHNSTrainer in trainers.py
from trainers import MDHNSTrainer 
from models import SASRec, Mamba4Rec, GRU4Rec, TTT4Rec, Narm, Linrec, LightSANs, FMLPRecModel
from utils import *
import time
time_stmp = time.time()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--output_dir', default='output/', type=str)
    parser.add_argument('--data_name', default='Beauty', type=str)
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--ckp', default=0, type=int, help="pretrain epochs 10, 20, 30...")
    
    # model args
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--embedding_size", type=int, default=64, help="embedding size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str) # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=50, type=int)
    
    # ▼▼▼ MD-HNS & DWS specific args ▼▼▼
    parser.add_argument('--backbone', type=str, default='SASRec', help="Backbone model 1 name (e.g., SASRec)")
    parser.add_argument('--backbone2', type=str, default='Mamba4Rec', help="Backbone model 2 name (e.g., Mamba4Rec)")
    parser.add_argument("--K_hns", type=int, default=10, help="Top-K for hard negative sampling pool")
    # ▼ DWS ▼ New hyperparameters for DWS and generalized loss
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for self-reflection loss term.")
    parser.add_argument("--beta", type=float, default=1.0, help="Weight for peer-guidance loss term.")
    parser.add_argument("--d_lambda", type=float, default=1.0, help="Weight for ensemble consensus loss term.")
    parser.add_argument("--dws_beta", type=float, default=0.5, help="Weight for the disagreement score in DWS.")
    parser.add_argument("--kd_temperature", type=float, default=1.0, help="Temperature for softmax in DWS.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for softmax in DWS.")
    parser.add_argument("--kd_gamma", type=float, default=1.0, help="蒸馏的比例")

    # ▲ DWS ▲
    # ▲▲▲ MD-HNS & DWS specific args ▲▲▲

    # Mamba block args
    parser.add_argument('--d_state', type=int, help='Dimension of state', default=16)
    parser.add_argument('--d_conv', type=int, help='Dimension of convolution', default=4)
    parser.add_argument('--expand', type=int, help='Expansion factor', default=2)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=1024, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--weight_decay", type=float, default=0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    # Neg sample (MD-HNS uses its own sampling during loss calculation, this defines the candidate pool)
    parser.add_argument("--neg_sampler", type=str, default="DNS", help="neg_sampler for candidate pool generation (DNS or Uniform)")
    parser.add_argument("--loss_type", type=str, default="BPR", help="Loss type, BPR or BCE")
    parser.add_argument("--N", type=int, default=200, help="candidate pool size") # N for DNSDataset

    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    # Data Loading
    args.data_file = args.data_dir + args.data_name
    train_data, max_item, _ = get_user_seqs(args.data_file + "_train.txt")
    valid_data, max_item_val, _ = get_user_seqs(args.data_file + "_val.txt")
    test_data, max_item_test, _ = get_user_seqs(args.data_file + "_test.txt")
    args.item_size = max(max_item, max_item_val, max_item_test) + 2
    
    print(f"Data loaded. Item size: {args.item_size}")
    
    # Generate rating matrix for filtering seen items during evaluation
    valid_rating_matrix = generate_rating_matrix(valid_data, args.item_size)
    test_rating_matrix = generate_rating_matrix(test_data, args.item_size)

    # Log and model save path
    # ▼ DWS ▼ Updated args_str to include new hyperparameters
    args_str = f'DWS-{args.backbone}-{args.backbone2}-{args.data_name}-a{args.alpha}-b{args.beta}-l{args.d_lambda}-dwsb{args.dws_beta}-t{args.temperature}'
    # ▲ DWS ▲
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    checkpoint = args_str + '-' + str(time_stmp) + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    print(str(args))
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    # Datasets and Dataloaders
    neg_sampler_dict = {'Uniform': SRDataset, "DNS": DNSDataset}
    train_dataset = neg_sampler_dict[args.neg_sampler](args, train_data)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=8)

    eval_dataset = neg_sampler_dict[args.neg_sampler](args, valid_data)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size,num_workers=2)

    test_dataset = neg_sampler_dict[args.neg_sampler](args, test_data)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size,num_workers=2)

    # Model Creation
    model_dict = {
        'SASRec': SASRec, 'Mamba4Rec': Mamba4Rec, 'GRU4Rec': GRU4Rec,
        'TTT4Rec': TTT4Rec, 'Narm': Narm, 'Linrec': Linrec,
        'LightSANs': LightSANs, 'FMLPRecModel': FMLPRecModel
    }
    
    model1 = model_dict[args.backbone](args=args)
    model2 = model_dict[args.backbone2](args=args)
    
    # Use MDHNSTrainer
    trainer = MDHNSTrainer(model1, model2, train_dataloader, eval_dataloader, test_dataloader, args)

    if args.do_eval:
        trainer.load(args.checkpoint_path)
        print(f'Load models from {args.checkpoint_path} for test!')
        trainer.args.train_matrix = test_rating_matrix
        _, result_infos = trainer.test(0, full_sort=True)
        print("Final Test Results:")
        print(f"Model 1 ({args.backbone}): {result_infos[0]}")
        print(f"Model 2 ({args.backbone2}): {result_infos[1]}")
        print(f"Ensemble: {result_infos[2]}")
    else:
        early_stopping = EarlyStopping(args.checkpoint_path, patience=30, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            
            print('\n' + '---'*10 + f' Validation for Epoch {epoch} ' + '---'*10)
            trainer.args.train_matrix = valid_rating_matrix
            # valid returns (ensemble_scores, [all_model_scores_list])
            sasrec_scores, _ = trainer.valid(epoch, full_sort=True)
            
            # Using ensemble model's NDCG@20 for Early Stopping
            # ensemble_scores format: [recall@5, ndcg@5, recall@10, ndcg@10, recall@20, ndcg@20]
            current_metric = [sasrec_scores[5]] 
            print('\n' + '---'*10 + f' Test for Epoch {epoch} ' + '---'*10)
            trainer.args.train_matrix = test_rating_matrix
            sasrec_scores, all_scores_list = trainer.test(epoch, full_sort=True)
            early_stopping(current_metric, trainer, all_scores_list) 

            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break
        
        print('\n' + '---'*10 + ' Best Validation Scores for Test ' + '---'*10)
        print(early_stopping.all_scores[0])

if __name__ == "__main__":
    main()