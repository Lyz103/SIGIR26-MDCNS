import random

import torch
from torch.utils.data import Dataset
import torch.nn as nn
from utils import neg_sample, neg_sample_dns_unique
import torch.nn.functional as F





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



class base_sampler(nn.Module):
    """
    Uniform sampler
    """
    def __init__(self, num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs):
        super(base_sampler, self).__init__()
        self.num_items = num_items
        self.num_neg = num_neg
        self.device = device
    
    def update_pool(self, model, **kwargs):
        pass
    
    def forward(self, user_id, **kwargs):
        batch_size = user_id.shape[0]
        return torch.randint(0, self.num_items, size=(batch_size, self.num_neg), device=self.device), -torch.log(self.num_items * torch.ones(batch_size, self.num_neg, device=self.device))

class two_pass(base_sampler):
    def __init__(self, num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs):
        super().__init__(num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs)
        self.num_users = num_users
        self.sample_size = sample_size # importance sampling
        self.pool_size = pool_size # resample
        self.pool = torch.zeros(num_users, pool_size, device=device, dtype=torch.long)
    
    def update_pool(self, model, input_ids, batch_idx, batch_size=1024, cover_flag=False, **kwargs):
        neg_items, neg_q = self.sample_Q(batch_idx)
        tmp_pool, tmp_score = self.re_sample(input_ids, model, neg_items, neg_q)
        self.__update_pool__(batch_idx, tmp_pool, tmp_score, cover_flag=cover_flag)   
    def sample_Q(self, user_batch):
        batch_size = user_batch.shape[0]
        return torch.randint(0, self.num_items, size=(batch_size, self.sample_size), device=self.device), -torch.log(self.num_items * torch.ones(batch_size, self.sample_size, device=self.device))
    
    def re_sample(self, input_ids, model, neg_items, log_neg_q):
        # print("log_neg_q", log_neg_q.shape)
        sequence_output = model.finetune(input_ids)[:, -1, :]
        test_item_emb = model.item_embeddings.weight[neg_items]
        # print("sequence_output.shape", sequence_output.shape)
        # print("test_item_emb.shape", test_item_emb.shape)
        ratings = torch.matmul(
            sequence_output.unsqueeze(1),    # [batch_size, 1, hidden_size]
            test_item_emb.transpose(1, 2)    # [batch_size, hidden_size, num_neg]
        ).squeeze(1)  # -> [batch_size, num_neg]
        pred = ratings - log_neg_q
        sample_weight = F.softmax(pred, dim=-1)
        idices = torch.multinomial(sample_weight, self.pool_size, replacement=True)
        return torch.gather(neg_items, 1, idices), torch.gather(sample_weight, 1, idices)
    
    def __update_pool__(self, user_batch, tmp_pool, tmp_score, cover_flag):
        if cover_flag is True:
            self.pool[user_batch] = tmp_pool
            return
        
        idx = self.pool[user_batch].sum(-1) < 1
        
        user_init = user_batch[idx]
        self.pool[user_init] = tmp_pool[idx]

        user_update = user_batch[~idx]
        num_user_update = user_update.shape[0]
        idx_k = torch.randint(0, 2*self.pool_size, size=(num_user_update, self.pool_size), device=self.device)
        candidate = torch.cat([self.pool[user_update], tmp_pool[~idx]], dim=1)
        self.pool[user_update] = torch.gather(candidate, 1, idx_k)
        return
    
    # @profile
    def forward(self, user_id, **kwargs):
        batch_size = user_id.shape[0]
        candidates = self.pool[user_id]
        idx_k = torch.randint(0, self.pool_size, size=(batch_size, self.num_neg), device=self.device)
        return torch.gather(candidates, 1, idx_k), -torch.log(self.pool_size * torch.ones(batch_size, self.num_neg, device=self.device))



class two_pass_weight(two_pass):
    def __init__(self, num_users, num_items, sample_size, pool_size, num_neg, device, **kwargs):
        super(two_pass_weight, self).__init__(num_users, num_items, sample_size, pool_size, num_neg, device)
        self.pool_weight = torch.zeros(num_users, pool_size, device=device)
    
    def __update_pool__(self, user_batch, tmp_pool, tmp_score, cover_flag=False):
        if cover_flag is True:
            self.pool[user_batch] = tmp_pool
            self.pool_weight[user_batch] = tmp_score.detach()
            return

        idx = self.pool[user_batch].sum(-1) < 1
        
        user_init = user_batch[idx]
        if len(user_init) > 0:
            self.pool[user_init] = tmp_pool[idx]
            self.pool_weight[user_init] = tmp_score[idx]

        user_update = user_batch[~idx]
        num_user_update = user_update.shape[0]
        if num_user_update > 0:
            idx_k = torch.randint(0, 2*self.pool_size, size=(num_user_update, self.pool_size), device=self.device)
            candidate = torch.cat([self.pool[user_update], tmp_pool[~idx]], dim=1)
            candidate_weight = torch.cat([self.pool_weight[user_update], tmp_score[~idx]], dim=1)
            self.pool[user_update] = torch.gather(candidate, 1, idx_k)
            self.pool_weight[user_update] = torch.gather(candidate_weight, 1, idx_k).detach()
    
    def forward(self, user_id, **kwargs):
        batch_size = user_id.shape[0]
        candidates = self.pool[user_id]
        candidates_weight = self.pool_weight[user_id]
        idx_k = torch.randint(0, self.pool_size, size=(batch_size, self.num_neg), device=self.device)
        return torch.gather(candidates, 1, idx_k), -torch.log(torch.gather(candidates_weight, 1, idx_k))