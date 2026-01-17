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
from models import SASRec, GRU4Rec, TTT4Rec, Narm, Linrec, LightSANs, FMLPRecModel
from utils import *
import time
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_geometric.utils import to_dense_adj
import dgl
time_stmp = time.time()


def get_single_gnno_graph(datapath):
    seq_list, _, _ = get_user_seqs(datapath + "_train.txt")
    unrepeat_seqs = [np.array(seq) for seq in seq_list]
    unrepeat_dict = {idx: seq for idx, seq in enumerate(unrepeat_seqs)}

    graph = construct_graph(unrepeat_dict)

    return graph


# GNNO构图
def build_WITG_from_trainset(dataset, use_renorm=True, use_scale=False, user_seq=False):
    seqs = []
    item_set = set()
    for record in dataset.items():
        if user_seq:
            items = record[1]
        else:
            items = record[1]['item_id']
        seqs.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)
    num_node = max_item + 1 # 有待考究，反正应该和item id对上就行

    relation = []
    adj = [dict() for _ in range(num_node)]

    for i in range(len(seqs)):
        data = seqs[i]
        for k in range(1, 4): # 只建立局部关系，(1,4)，再往后不会建立边了？
            for j in range(len(data) - k):
                relation.append([data[j], data[j + k], k])
                relation.append([data[j + k], data[j], k])
    for temp in relation: # 初始化还是更新
        if temp[1] in adj[temp[0]].keys():
            adj[temp[0]][temp[1]] += 1 / temp[2]
        else:
            adj[temp[0]][temp[1]] = 1 / temp[2]

    adj_pyg = [] # 节点表，排好序的
    weight_pyg = [] # 按照节点归一化后的边权值

    for t in range(1, num_node):
        x = [v for v in sorted(adj[t].items(), reverse=True, key=lambda x: x[1])]
        adj_pyg += [[t, v[0]] for v in x]
        if use_scale:
            t_sum = 0
            for v in x:
                t_sum += v[1]
            weight_pyg += [v[1] / t_sum for v in x]
        else:
            weight_pyg += [v[1] for v in x]

    adj_np = np.array(adj_pyg)
    adj_np = adj_np.transpose() # 起始，终点
    edge_np = np.array([adj_np[0, :], adj_np[1, :]])
    x = torch.arange(0, num_node).long().view(-1, 1)  # torch.int64[n_node, 1], item entity index
    edge_attr = torch.from_numpy(np.array(weight_pyg)).view(-1, 1)  # torch.float64[n_edge, 1]
    edge_index = torch.from_numpy(edge_np).long()  # torch.int64[2, n_edge]
    Graph_data = Data(x, edge_index, edge_attr=edge_attr)
    print(Graph_data)
    if use_renorm:
        row, col = Graph_data.edge_index[0], Graph_data.edge_index[1]
        row_deg = 1. / degree(row, num_node, Graph_data.edge_attr.dtype)
        col_deg = 1. / degree(col, num_node, Graph_data.edge_attr.dtype)
        deg = row_deg[row] + col_deg[col]
        new_att = edge_attr * deg.view(-1, 1) # 根据度的情况再归一化一下权重
        Graph_data.edge_attr = new_att

    # torch.save(Graph_data, datapath + 'witg.pt')
    return Graph_data


def construct_graph(inter_df):
    if not isinstance(inter_df, dict):
        # 传入交互的df吧，按照user，time排序
        inter_df = inter_df.sort_values(by=['user_id', 'time'])
        # 获得序列
        user_seqs = inter_df.groupby('user_id')['item_id'].apply(np.array).to_dict()
    else:
        user_seqs = inter_df
    gdata = build_WITG_from_trainset(user_seqs, use_renorm=True, use_scale=True, user_seq=True)
    edge_index = gdata.edge_index.detach().cpu().numpy()
    # 获得图对象
    graph_dgl = dgl.graph((edge_index[0], edge_index[1]))
    graph_dgl.edata['w'] = gdata.edge_attr
    #
    nodes = graph_dgl.nodes()
    # 获得子图
    ego_nodes = [dgl.khop_in_subgraph(graph_dgl, n, k=1) for n in tqdm(nodes)]
    # 获得邻居节点集合
    nei_nodes = [ego[0].dstdata['_ID'].cpu() for ego in tqdm(ego_nodes)]
    # 临界矩阵
    graph_adj = graph_dgl.adj().to_dense()

    # 计算jaccard
    g_cn = torch.matmul(graph_adj, graph_adj.t())  # 应该是计算共用节点的数量
    g_deg = g_cn.diag().expand_as(g_cn)
    g_union = g_deg + g_deg.t() - g_cn  # 计算邻居节点并集的数量
    g_jaccard = (g_cn / g_union).nan_to_num()  # 计算jaccard近似度

    print("graph constructed")

    return SequentialGraph(gdata,graph_dgl, nei_nodes, graph_adj, g_jaccard)

    # return (gdata, graph_dgl, nei_nodes, graph_adj, g_jaccard)  # 一个元组

class SequentialGraph:
    def __init__(self,gdata,graph_dgl, nei_nodes, graph_adj, g_jaccard):
        self.gdata = gdata
        self.graph_dgl = graph_dgl
        self.nei_nodes = nei_nodes
        self.graph_adj = graph_adj
        self.jaccard = g_jaccard

    # 先只放jaccard
    def to_device(self,device):
        self.jaccard = self.jaccard.to(device)




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
    parser.add_argument("--N", type=int, default=200, help="sample_size")
    parser.add_argument("--M", type=int, default=10, help="pool_size")
    parser.add_argument("--neg_sampler", type=str, default="DNS", help="neg_sampler")
    parser.add_argument("--loss_type", type=str, default="BPR", help="BCE,BPR,SFM")
    parser.add_argument("--CL_type", type=str, default="Radical", help="Radical, Gentle")
    parser.add_argument("--start_epoch", default=30, type=int)
    parser.add_argument("--K", default=0.05, type=int)




    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)


    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
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
    seq_graph = get_single_gnno_graph(args.data_file) # 子序列

    kwargs  = {
        'gnno':seq_graph,
        'gnums':10,
        'hardness':0,
    }
    hardness = min(0.01 * 5, 0.4)
    kwargs.update({
        'hardness': hardness
    })
    args.N = args.N - kwargs['gnums']
    print("neg_sampler: ", args.neg_sampler)
    args.kwargs = kwargs
    train_dataset = neg_sampler_dict[args.neg_sampler](args, train_data)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=20)

    eval_dataset = neg_sampler_dict[args.neg_sampler](args, valid_data)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    test_dataset = neg_sampler_dict[args.neg_sampler](args, test_data)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    # 创建模型名称到类的映射
    model_dict = {
        'SASRec': SASRec,
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
        # train
            if hardness < 0.4:
                hardness = hardness + 0.01
                hardness = min(hardness, 0.4)
                kwargs.update({
                    'hardness':hardness
                })
            trainer.args.kwargs = kwargs
            
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