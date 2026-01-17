# -*- coding: utf-8 -*-
# @Time    : 2020/3/30 11:06
# @Author  :

import numpy as np
import tqdm
import random
import time
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F # ▼ DWS ▼ Import F for softmax
from utils import recall_at_k, ndcg_k, get_metric, generate_scaled_fx




class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):

        self.args = args
        self.cuda_condition = True and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        self.criterion = nn.BCELoss()

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def _DNS(self, batch_neg_candidates, seq_out, M):
        """
        使用动态负采样（DNS）从候选负样本中选择
        
        Args:
            batch_neg_candidates: 形状为 [B, N] 的负样本候选项
            seq_out: 形状为 [B, hidden_size] 的序列输出
            model: 推荐模型
            M: 考虑的顶部候选项数量
        
        Returns:
            形状为 [B] 的选定负样本
        """
        batch_size, N = batch_neg_candidates.size()
        device = seq_out.device
        
        # 获取模型中的项目嵌入
        with torch.no_grad():
            item_emb = self.model.item_embeddings.weight
            
            # 为每个批次计算负样本的分数
            selected_neg = torch.zeros(batch_size, dtype=torch.long, device=device)
            
            for i in range(batch_size):
                # 获取当前序列的负样本候选项
                neg_candidates = batch_neg_candidates[i]  # [N]
                
                # 获取这些负样本候选项的嵌入
                neg_emb = item_emb[neg_candidates]  # [N, hidden_size]
                
                # 计算预测分数
                neg_scores = torch.matmul(seq_out[i].unsqueeze(0), neg_emb.transpose(0, 1)).squeeze(0)  # [N]
                
                # 获取得分最高的M个负样本
                _, top_indices = torch.topk(neg_scores, min(M, N))
                top_neg_candidates = neg_candidates[top_indices]
                
                # 从前M个中随机选择一个
                selected_idx = random.randint(0, min(M, N) - 1)
                selected_neg[i] = top_neg_candidates[selected_idx]
        
        return selected_neg
    

    def _random_neg_sampling(self, batch_neg_candidates, seq_out, M=None):
        """
        从候选负样本中随机选择一个
        
        Args:
            batch_neg_candidates: 形状为 [B, N] 的负样本候选项
            seq_out: 形状为 [B, hidden_size] 的序列输出
            M: 不使用，保留参数以保持接口一致
        
        Returns:
            形状为 [B] 的选定负样本
        """
        batch_size, N = batch_neg_candidates.size()
        device = seq_out.device
        
        # 为每个批次随机选择一个负样本
        selected_neg = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        for i in range(batch_size):
            # 获取当前序列的负样本候选项
            neg_candidates = batch_neg_candidates[i]  # [N]
            
            # 随机选择一个索引
            selected_idx = random.randint(0, N - 1)
            selected_neg[i] = neg_candidates[selected_idx]
        
        return selected_neg
        



    # length: [length_lower_bound, length_upper_bound)
    def get_sample_scores_length(self, epoch, answers, pred_list, original_input_length, length_lower_bound, length_upper_bound):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        filter_pred_list = []
        for i in range(len(original_input_length)):  # length filter
            if length_lower_bound <= original_input_length[i] and original_input_length[i] < length_upper_bound:
                filter_pred_list.append(pred_list[i])
        pred_list = np.array(filter_pred_list)
        R_5, NDCG_5, MRR_5 = get_metric(pred_list, 5)
        R_10, NDCG_10, MRR_10 = get_metric(pred_list, 10)
        R_20, NDCG_20, MRR_20 = get_metric(pred_list, 20)

        post_fix = {
            "Epoch": epoch,
            "HR_5": '{:.7f}'.format(R_5), "HR_10": '{:.7f}'.format(R_10), "HR_20": '{:.7f}'.format(R_20),
            "NDCG@5": '{:.7f}'.format(NDCG_5), "NDCG@10": '{:.7f}'.format(NDCG_10), "NDCG@20": '{:.7f}'.format(NDCG_20),
            "MRR@5": '{:.7f}'.format(MRR_5), "MRR@10": '{:.7f}'.format(MRR_10), "MRR@20": '{:.7f}'.format(MRR_20)
        }
        print(str(length_lower_bound) + " " + str(post_fix))
        with open(self.args.log_file, 'a') as f:
            f.write(str(length_lower_bound) + " " + str(post_fix) + '\n')
        return str(post_fix)

    def get_sample_scores(self, epoch, answers, pred_list, original_input_length):
        length_lower_bound = [0, 20, 30, 40]
        length_upper_bound = [20, 30, 40, 51]
        for i in range(len(length_lower_bound)):
            self.get_sample_scores_length(epoch, answers, pred_list, original_input_length, length_lower_bound[i], length_upper_bound[i])
        # print(post_fix)
        # with open(self.args.log_file, 'a') as f:
        #     f.write(str(post_fix) + '\n')
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        # HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        # R_20 = recall_at_k(answers, pred_list, 20)
        # R_50 = recall_at_k(answers, pred_list, 50)
        R_5, NDCG_5, MRR_5 = get_metric(pred_list, 5)
        R_10, NDCG_10, MRR_10 = get_metric(pred_list, 10)
        R_20, NDCG_20, MRR_20 = get_metric(pred_list, 20)

        post_fix = {
            "Epoch": epoch,
            "HR_5": '{:.7f}'.format(R_5), "HR_10": '{:.7f}'.format(R_10), "HR_20": '{:.7f}'.format(R_20),
            "NDCG@5": '{:.7f}'.format(NDCG_5), "NDCG@10": '{:.7f}'.format(NDCG_10), "NDCG@20": '{:.7f}'.format(NDCG_20),
            "MRR@5": '{:.7f}'.format(MRR_5), "MRR@10": '{:.7f}'.format(MRR_10), "MRR@20": '{:.7f}'.format(MRR_20)
        }
        return [R_5, R_10, R_20, NDCG_5, NDCG_10, NDCG_20, MRR_5, MRR_10, MRR_20], str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HIT@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def cross_entropy(self,seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)

        # [batch hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(1))
        neg = neg_emb.view(-1, neg_emb.size(1))
        seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0)).float()  # [batch]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget) 

        return loss
    
    def bpr_loss(self, seq_out, pos_ids, neg_ids):
        """
        实现BPR (Bayesian Personalized Ranking) Loss
        
        Args:
            seq_out: 序列输出, [batch, hidden_size]
            pos_ids: 正样本ID, [batch]
            neg_ids: 负样本ID, [batch]
            
        Returns:
            loss: BPR损失值
        """
        # 获取正样本和负样本的嵌入
        pos_emb = self.model.item_embeddings(pos_ids)  # [batch, hidden_size]
        neg_emb = self.model.item_embeddings(neg_ids)  # [batch, hidden_size]
        
        # 计算序列输出和物品嵌入的内积
        pos_logits = torch.sum(pos_emb * seq_out, -1)  # [batch]
        neg_logits = torch.sum(neg_emb * seq_out, -1)  # [batch]
        
        # 确定哪些位置是有效的（pos_ids > 0表示有效位置）
        istarget = (pos_ids > 0).float()  # [batch]
        
        # 计算BPR损失: -log(sigmoid(pos_logits - neg_logits))
        # 只考虑有效位置的损失
        loss = -torch.log(torch.sigmoid(pos_logits - neg_logits) + 1e-24) * istarget
        
        # 对有效位置的损失取平均
        loss = torch.sum(loss) / torch.sum(istarget)
        
        return loss

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


class FinetuneTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(FinetuneTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )
        if self.args.loss_type=="BCE":
            self.loss = self.cross_entropy
        elif self.args.loss_type=="BPR":
            self.loss = self.bpr_loss
        
    def iteration(self, epoch, dataloader, full_sort=False, train=True):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar

        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}", colour="#00ff00")
        if train:
            self.model.train()
            avg_loss = 0.0
            rec_avg_loss = 0.0
            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)
                user_id, input_ids, target_pos, target_neg, _, _ = batch
                # Binary cross_entropy

                sequence_output = self.model.finetune(input_ids)[:, -1, :]

                if self.args.neg_sampler=="DNS":
                    target_neg = self._DNS(target_neg, sequence_output, self.args.M)
                elif self.args.neg_sampler=="Uniform":
                    target_neg = target_neg

                loss = self.loss(sequence_output, target_pos, target_neg) 
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                avg_loss += loss.item()

            post_fix = {
                "epoch": epoch,
                "loss": '{:.4f}'.format(avg_loss / len(rec_data_iter))
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, _, answers, _ = batch

                    recommend_output = self.model.finetune(input_ids)[:, -1, :]

                    # 推荐的结果
                    rating_pred = self.predict_full(recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                    # argpartition 时间复杂度O(n)  argsort O(nlogn) 只会做
                    # 加负号"-"表示取大的值
                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    # 根据返回的下标 从对应维度分别取对应的值 得到每行topk的子表
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    # 对子表进行排序 得到从大到小的顺序
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    # 再取一次 从ind中取回 原来的下标
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list)



# =====================================================================================
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 修改后的 MD-HNS + KD Trainer ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# =====================================================================================
class MDHNSTrainer(Trainer):
    def __init__(self, model1, model2, train_dataloader, eval_dataloader, test_dataloader, args):
        # ... (init 的前半部分保持不变) ...
        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")
        self.model1 = model1
        self.model2 = model2
        if self.cuda_condition:
            self.model1.cuda()
            self.model2.cuda()
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim1 = Adam(self.model1.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)
        self.optim2 = Adam(self.model2.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)
        print(f"Total Parameters (Model 1 - {self.args.backbone}):", sum([p.nelement() for p in self.model1.parameters()]))
        print(f"Total Parameters (Model 2 - {self.args.backbone2}):", sum([p.nelement() for p in self.model2.parameters()]))

        # ▼ 优化 (分离式) ▼ 指向新的、分离式损失函数
        if self.args.loss_type.upper() == "BPR":
            self.loss_function = self._bpr_loss_separate
            print("Using BPR loss (Candidate-Set Optimized, Separate) for training.")
        elif self.args.loss_type.upper() == "BCE":
            self.loss_function = self._bce_loss_separate
            print("Using BCE loss (Candidate-Set Optimized, Separate) for training.")
        else:
            raise ValueError(f"Unsupported loss_type: {self.args.loss_type}. Please choose 'BPR' or 'BCE'.")

        if self.args.kd_gamma > 0:
            print(f"Using Collaborative KD (Candidate-Set Optimized) with gamma={self.args.kd_gamma}, T={self.args.kd_temperature}")
        else:
            print("Knowledge Distillation is OFF (kd_gamma = 0).")
    
    # ▼ 优化 (分离式) ▼ 新的损失函数，接收分离的 pos_logits [B] 和 neg_logits_pool [B, N]
    def _bce_loss_separate(self, pos_logits, neg_logits_pool, neg_indices, istarget):
        """
        Calculates pairwise BCE loss from separate pos/neg logits.
        """
        # pos_logits is [B]
        # neg_logits_pool is [B, N], neg_indices is [B] (relative index 0..N-1)
        neg_scores = neg_logits_pool.gather(1, neg_indices.unsqueeze(1)).squeeze(1) # [B]

        pos_loss = -torch.log(torch.sigmoid(pos_logits) + 1e-24)
        neg_loss = -torch.log(1 - torch.sigmoid(neg_scores) + 1e-24)

        loss = (pos_loss + neg_loss) * istarget
        return torch.sum(loss) / (torch.sum(istarget) + 1e-8)

    # ▼ 优化 (分离式) ▼ 新的损失函数，接收分离的 pos_logits [B] 和 neg_logits_pool [B, N]
    def _bpr_loss_separate(self, pos_logits, neg_logits_pool, neg_indices, istarget):
        """
        Calculates BPR loss from separate pos/neg logits.
        """
        # pos_logits is [B]
        # neg_logits_pool is [B, N], neg_indices is [B] (relative index 0..N-1)
        neg_scores = neg_logits_pool.gather(1, neg_indices.unsqueeze(1)).squeeze(1) # [B]
        
        loss = -torch.log(torch.sigmoid(pos_logits - neg_scores) + 1e-24) * istarget
        return torch.sum(loss) / (torch.sum(istarget) + 1e-8)

    # ▼ 优化 (分离式) ▼ HNS 函数保持不变 (它已在 [B, N] 上操作)
    def _dws_hns_optimized(self, neg_logits1, neg_logits2, K, dws_beta):
        """
        Optimized DWS-HNS operating on negative candidate logits [B, N].
        Returns relative indices [B] (from 0 to N-1).
        """
        with torch.no_grad():
            item_disagreement = torch.abs(neg_logits1 - neg_logits2) # [B, N]

            def rerank_and_sample(logits, disagreement, K, dws_beta):
                batch_size, N = logits.size()
                enhanced_scores = logits + dws_beta * disagreement # [B, N]
                _, top_indices = torch.topk(enhanced_scores, min(K, N)) # [B, K]
                random_choice = torch.randint(0, top_indices.size(1), (batch_size,)).to(self.device) # [B]
                chosen_topk_indices = top_indices[torch.arange(batch_size), random_choice] # [B]
                return chosen_topk_indices

            logits_ens = 0.5 * (neg_logits1 + neg_logits2)
            
            neg_A_idx = rerank_and_sample(neg_logits1, item_disagreement, K, dws_beta)
            neg_B_idx = rerank_and_sample(neg_logits2, item_disagreement, K, dws_beta)
            neg_C_idx = rerank_and_sample(logits_ens, item_disagreement, K, dws_beta)

        return neg_A_idx, neg_B_idx, neg_C_idx
    # ▲ 优化 (分离式) ▲

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        str_code = "train" if train else "valid" if self.test_dataloader!=dataloader else "test"
        
        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                    desc=f"DWS-HNS EP_{str_code}:{epoch}",
                                    total=len(dataloader),
                                    bar_format="{l_bar}{r_bar}", colour="#00ff00")
        
        # =================================================================
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 优化后的训练循环 (分离式) ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # =================================================================
        if train:
            self.model1.train()
            self.model2.train()
            
            avg_loss = 0.0
            avg_bpr_loss = 0.0
            avg_kd_loss = 0.0

            for i, batch in rec_data_iter:
                batch = tuple(t.to(self.device) for t in batch)
                _, input_ids, target_pos, target_neg_candidates, _, _ = batch
                
                istarget = (target_pos > 0).float()
                batch_size = input_ids.size(0)

                # 1. Get sequence outputs
                seq_out1 = self.model1.finetune(input_ids)[:, -1, :] # [B, H]
                seq_out2 = self.model2.finetune(input_ids)[:, -1, :] # [B, H]
                
                # 2. ▼ 优化 (分离式) ▼ 分别获取正例和负例的嵌入
                # (target_pos [B] -> [B, 1] -> [B, 1, H])
                pos_emb1 = self.model1.item_embeddings(target_pos.unsqueeze(1))
                pos_emb2 = self.model2.item_embeddings(target_pos.unsqueeze(1))
                # (target_neg_candidates [B, N] -> [B, N, H])
                neg_emb1 = self.model1.item_embeddings(target_neg_candidates)
                neg_emb2 = self.model2.item_embeddings(target_neg_candidates)

                # 3. ▼ 优化 (分离式) ▼ 分别计算正例和负例的 logits
                # (B, 1, H) * (B, 1, H) -> sum -> (B, 1) -> squeeze -> [B]
                pos_logits1 = (seq_out1.unsqueeze(1) * pos_emb1).sum(dim=-1).squeeze(1)
                pos_logits2 = (seq_out2.unsqueeze(1) * pos_emb2).sum(dim=-1).squeeze(1)
                # (B, 1, H) * (B, N, H) -> sum -> [B, N]
                neg_logits1_pool = (seq_out1.unsqueeze(1) * neg_emb1).sum(dim=-1)
                neg_logits2_pool = (seq_out2.unsqueeze(1) * neg_emb2).sum(dim=-1)

                # 4. ▼ 优化 (分离式) ▼ HNS 在 [B, N] 的 logits_pool 上操作
                neg_A_idx, neg_B_idx, neg_C_idx = self._dws_hns_optimized(
                    neg_logits1_pool, 
                    neg_logits2_pool, 
                    self.args.K_hns,
                    self.args.dws_beta
                )
                
                # 5. ▼ 优化 (分离式) ▼ BPR/BCE 损失
                # L^{(1)}
                loss1_A = self.loss_function(pos_logits1, neg_logits1_pool, neg_A_idx, istarget) # Self
                loss1_B = self.loss_function(pos_logits1, neg_logits1_pool, neg_B_idx, istarget) # Peer
                loss1_C = self.loss_function(pos_logits1, neg_logits1_pool, neg_C_idx, istarget) # Ens
                L1 = self.args.alpha * loss1_A + self.args.beta * loss1_B + self.args.d_lambda * loss1_C
                
                # L^{(2)}
                loss2_A = self.loss_function(pos_logits2, neg_logits2_pool, neg_A_idx, istarget) # Peer
                loss2_B = self.loss_function(pos_logits2, neg_logits2_pool, neg_B_idx, istarget) # Self
                loss2_C = self.loss_function(pos_logits2, neg_logits2_pool, neg_C_idx, istarget) # Ens
                L2 = self.args.alpha * loss2_B + self.args.beta * loss2_A + self.args.d_lambda * loss2_C
                
                total_bpr_loss = L1 + L2
                
                # 6. ▼ 优化 (分离式) ▼ KD 损失 (在计算时再拼接)
                total_kd_loss = 0.0
                if self.args.kd_gamma > 0:
                    # 6.1. 为 KD 拼接 [B, N+1] 的 logits
                    candidate_logits1 = torch.cat([pos_logits1.unsqueeze(1), neg_logits1_pool], dim=1) # [B, N+1]
                    candidate_logits2 = torch.cat([pos_logits2.unsqueeze(1), neg_logits2_pool], dim=1) # [B, N+1]
                    
                    logits_ens_candidates = 0.5 * (candidate_logits1 + candidate_logits2) # [B, N+1]
                    
                    # 6.2. 计算 KD 损失
                    with torch.no_grad():
                        p_teacher = F.softmax(logits_ens_candidates.detach() / self.args.kd_temperature, dim=-1)
                    
                    log_p_student1 = F.log_softmax(candidate_logits1 / self.args.kd_temperature, dim=-1)
                    log_p_student2 = F.log_softmax(candidate_logits2 / self.args.kd_temperature, dim=-1)
                    
                    T_squared = self.args.kd_temperature ** 2
                    
                    L_KD1 = T_squared * (F.kl_div(log_p_student1, p_teacher, reduction='sum'))
                    L_KD2 = T_squared * (F.kl_div(log_p_student2, p_teacher, reduction='sum'))
                    
                    total_kd_loss = self.args.kd_gamma * (L_KD1 + L_KD2)

                # 7. 总损失 (BPR + KD)
                total_loss = total_bpr_loss + total_kd_loss

                # 8. 优化
                self.optim1.zero_grad()
                self.optim2.zero_grad()
                total_loss.backward()
                self.optim1.step()
                self.optim2.step()
                
                # 9. 累积损失
                avg_loss += total_loss.item()
                avg_bpr_loss += total_bpr_loss.item()
                avg_kd_loss += total_kd_loss.item() if isinstance(total_kd_loss, torch.Tensor) else total_kd_loss
            
            # ... (post_fix 和日志记录保持不变) ...
            post_fix = {
                "epoch": epoch,
                "avg_loss": '{:.4f}'.format(avg_loss / len(rec_data_iter)),
                "avg_bpr_loss": '{:.4f}'.format(avg_bpr_loss / len(rec_data_iter)),
                "avg_kd_loss": '{:.4f}'.format(avg_kd_loss / len(rec_data_iter))
            }
            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))
            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')
        
        # =================================================================
        # ▼▼▼▼▼▼▼ 评估/测试循环 (保持不变, 仍需 FULL LOGITS) ▼▼▼▼▼▼▼
        # =================================================================
        else: # Validation / Testing
            # ... (评估/测试部分的代码与之前完全相同) ...
            self.model1.eval()
            self.model2.eval()
            pred_list1, pred_list2, pred_list_ens = None, None, None
            answer_list = None
            for i, batch in rec_data_iter:
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, _, _, answers, _ = batch
                with torch.no_grad():
                    seq_out1 = self.model1.finetune(input_ids)[:, -1, :]
                    seq_out2 = self.model2.finetune(input_ids)[:, -1, :]
                    logits1 = torch.matmul(seq_out1, self.model1.item_embeddings.weight.transpose(0, 1))
                    logits2 = torch.matmul(seq_out2, self.model2.item_embeddings.weight.transpose(0, 1))
                    logits_ens = 0.5 * (logits1 + logits2)
                batch_pred_list1 = self._get_ranked_list(logits1, user_ids)
                batch_pred_list2 = self._get_ranked_list(logits2, user_ids)
                batch_pred_list_ens = self._get_ranked_list(logits_ens, user_ids)
                if i == 0:
                    pred_list1, pred_list2, pred_list_ens = batch_pred_list1, batch_pred_list2, batch_pred_list_ens
                    answer_list = answers.cpu().data.numpy()
                else:
                    pred_list1 = np.append(pred_list1, batch_pred_list1, axis=0)
                    pred_list2 = np.append(pred_list2, batch_pred_list2, axis=0)
                    pred_list_ens = np.append(pred_list_ens, batch_pred_list_ens, axis=0)
                    answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
            scores1, info1 = self.get_full_sort_score(epoch, answer_list, pred_list1)
            scores2, info2 = self.get_full_sort_score(epoch, answer_list, pred_list2)
            scores_ens, info_ens = self.get_full_sort_score(epoch, answer_list, pred_list_ens)
            print(f"\nModel 1 ({self.args.backbone}) Results: {info1}")
            print(f"Model 2 ({self.args.backbone2}) Results: {info2}")
            print(f"Ensemble Results: {info_ens}")
            with open(self.args.log_file, 'a') as f:
                    f.write(f"Model 1 ({self.args.backbone}) Results: {info1}\n")
                    f.write(f"Model 2 ({self.args.backbone2}) Results: {info2}\n")
                    f.write(f"Ensemble Results: {info_ens}\n")
            return scores1, (info1, info2, info_ens)

    # ... (_get_ranked_list, save, load, get_full_sort_score 等方法保持不变) ...
    def _get_ranked_list(self, logits, user_ids):
        # ... (This helper method remains the same) ...
        rating_pred = logits.cpu().data.numpy().copy()
        batch_user_index = user_ids.cpu().numpy()
        if self.args.train_matrix is not None:
                rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = -np.inf
        
        ind = np.argpartition(rating_pred, -20)[:, -20:]
        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
        arr_ind_argsort = np.argsort(arr_ind, axis=1)[:, ::-1]
        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]
        return batch_pred_list