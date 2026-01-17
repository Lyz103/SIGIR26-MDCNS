import random

import torch
from torch.utils.data import Dataset

from utils import neg_sample, neg_sample_dns_unique

import numpy as np




class SRDataset(Dataset):   # 为了方便计算各个长度的指标，在dataset 中加入了 original_input_ids 表示原有序列的长度

    def __init__(self, args, user_seq, test_neg_items=None):
        self.args = args
        self.user_seq = user_seq
        self.max_len = args.max_seq_length
        self.test_neg_items = test_neg_items

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]
        input_ids = items[:-1]
        original_input_length = len(input_ids)
        answer = [items[-1]]

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = items[-1]
        target_neg = (neg_sample(answer, self.args.item_size))


        input_ids = input_ids[-self.max_len:]


        assert len(input_ids) == self.max_len



        cur_tensors = (
            torch.tensor(user_id, dtype=torch.long),  # user_id for testing
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long),
            torch.tensor(target_neg, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
            torch.tensor(original_input_length, dtype=torch.long),
        )

        return cur_tensors

    def __len__(self):
        return len(self.user_seq)
    



class DNSDataset(Dataset):   # 为了方便计算各个长度的指标，在dataset 中加入了 original_input_ids 表示原有序列的长度

    def __init__(self, args, user_seq, test_neg_items=None):
        self.args = args
        self.user_seq = user_seq
        self.max_len = args.max_seq_length
        self.test_neg_items = test_neg_items
        self.N = args.N

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]
        input_ids = items[:-1]
        original_input_length = len(input_ids)
        answer = [items[-1]]

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = items[-1]
        target_neg = neg_sample_dns_unique(answer, input_ids, self.args.item_size, self.N)


        input_ids = input_ids[-self.max_len:]


        assert len(input_ids) == self.max_len



        cur_tensors = (
            torch.tensor(user_id, dtype=torch.long),  # user_id for testing
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long),
            torch.tensor(target_neg, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
            torch.tensor(original_input_length, dtype=torch.long),
        )

        return cur_tensors

    def __len__(self):
        return len(self.user_seq)
    


# class SingleDomainBaseDataset(Dataset):
#     # 基本的dataset，不会采样负例，eval和test可以用
#     # item_num在这里目前没什么作用
#     def __init__(self,data_list,max_len,item_num=100):
#         super(SingleDomainBaseDataset, self).__init__()
#         self.dataset = data_list
#         self.max_len = max_len
#         self.item_num = item_num

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         seq = self.dataset[idx]
#         # 划分
#         target = seq[-1]
#         input_seq = seq[:-1]

#         # 切
#         if len(input_seq) > self.max_len:
#             input_seq = input_seq[-self.max_len:]
#         seq_len = len(input_seq)
#         # pad
#         pad_len = self.max_len - seq_len
#         input_seq = input_seq + [0] * pad_len

#         return input_seq,seq_len,target

# class SingleDomainMultiNegSRNSDataset(SingleDomainBaseDataset):
#     def __init__(self,data_list,max_len,item_num,candi_num,idstart=1):
#         super().__init__(data_list, max_len, item_num)

#         self.idstart = idstart
#         self.candi_num = candi_num
#         self.candi_set = dict()

#         self.init_candi_sampling()

#     def init_candi_sampling(self):
#         # 一开始就为每一条数据采样
#         # dict索引
#         total_items = list(range(self.idstart,self.idstart+self.item_num))
#         total_set = set(total_items)
#         for idx,data in enumerate(self.dataset):
#             oklist = list(total_set - set(data)) # 可行域
#             assert len(oklist) >= self.candi_num
#             temp = random.sample(oklist,self.candi_num)
#             self.candi_set[idx] = temp[:]

#     def __getitem__(self, idx):
#         input_seq, seq_len, target = super().__getitem__(idx)
#         negs = self.candi_set[idx]
#         answers = [target]
#         cur_tensors = (
#             torch.tensor(idx, dtype=torch.long),  # user_id for testing
#             torch.tensor(input_seq, dtype=torch.long),
#             torch.tensor(target, dtype=torch.long),
#             torch.tensor(negs, dtype=torch.long),
#             torch.tensor(answers, dtype=torch.long),
#             torch.tensor(seq_len, dtype=torch.long),
#         )

#         return cur_tensors