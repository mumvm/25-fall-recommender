'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import world
import torch
from torch import nn, optim
import numpy as np
from torch import log
from dataloader import BasicDataset
from time import time
from model import LightGCN
from model import PairWiseModel
from sklearn.metrics import roc_auc_score
from optimizers import ClusterCoupledAdam, ClusterCoupledSGD, ClusterCoupledRMSProp, Lamb 
import random
import os
import csv
try:
    from cppimport import imp_from_filepath
    from os.path import join, dirname
    path = join(dirname(__file__), "sources/sampling.cpp")
    sampling = imp_from_filepath(path)
    sampling.seed(world.seed)
    sample_ext = True
except:
    world.cprint("Cpp extension not loaded")
    sample_ext = False

class BPRLoss:
    def __init__(self,
                 recmodel: PairWiseModel,
                 config: dict):
        """
        BPRLoss: LightGCN 학습용 손실 + 옵티마이저 래퍼

        - config['optimizer'] == 'adam'   → 기존 Adam 사용
        - config['optimizer'] == 'cluster' → Cluster-Coupled AdamW 사용

        weight_decay(reg)는 논문/원 코드처럼
        self.model.bpr_loss에서 나온 reg_loss * config['decay']로만 적용하고,
        옵티마이저쪽 weight_decay는 0으로 두는 방향으로 맞춤.
        """
        self.model = recmodel
        self.weight_decay = config['decay']
        self.lr = config['lr']

        optimizer_name = config.get('optimizer', 'adam')

        if optimizer_name == 'cluster':
            # LightGCN의 임베딩 파라미터(embedding_user, embedding_item)에만
            # 클러스터링을 적용하고, 나머지는 일반 Adam처럼 사용
            emb_params = []
            other_params = []
            for name, p in self.model.named_parameters():
                if ('embedding_user' in name) or ('embedding_item' in name):
                    emb_params.append(p)
                else:
                    other_params.append(p)

            param_groups = []
            if emb_params:
                param_groups.append({
                    "params": emb_params,
                    "lr": self.lr,
                    "weight_decay": 0.0,  # reg_loss로만 decay 처리
                    "clustered": True,
                    "num_clusters": config.get("num_clusters", 16),
                    "alpha": config.get("alpha", 0.5),
                    "recluster_interval": config.get("recluster_interval", 100),
                })
            if other_params:
                param_groups.append({
                    "params": other_params,
                    "lr": self.lr,
                    "weight_decay": 0.0,  # 여기엔 클러스터링 X, 순수 Adam 업데이트
                })

            self.opt = ClusterCoupledAdam(
                param_groups,
                lr=self.lr,
                weight_decay=0.0,  # reg_loss 사용하므로 여기선 0
            )
        elif optimizer_name == 'cluster_sgd':
            emb_params = []
            other_params = []

            for name, p in self.model.named_parameters():
                if ('embedding_user' in name) or ('embedding_item' in name):
                    emb_params.append(p)
                else:
                    other_params.append(p)

            param_groups = []

            # cluster + SGD
            if emb_params:
                param_groups.append({
                    "params": emb_params,
                    "lr": self.lr,
                    "weight_decay": 0.0,
                    "clustered": True,
                    "num_clusters": config.get("num_clusters", 16),
                    "alpha": config.get("alpha", 0.5),
                    "recluster_interval": config.get("recluster_interval", 100),
                    "momentum": 0.9,
                })

            if other_params:
                param_groups.append({
                    "params": other_params,
                    "lr": self.lr,
                    "weight_decay": 0.0,
                    "momentum": 0.9,
                })

            self.opt = ClusterCoupledSGD(param_groups)

        elif optimizer_name == 'lamb':
            self.opt = Lamb(
                recmodel.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=config.get('lamb_betas', (0.9, 0.999)),
                eps=config.get('lamb_eps', 1e-6),
                clamp_value=config.get('lamb_clamp', 10),
                debias=config.get('lamb_debias', True),
            )

        elif optimizer_name == 'sgd':
            self.opt = optim.SGD(
                recmodel.parameters(),
                lr=self.lr,
                momentum=0.9
            )
        elif optimizer_name == "rmsprop":
          self.opt = optim.RMSprop(
              recmodel.parameters(),
              lr=self.lr,
              weight_decay=0.0
          )

        elif optimizer_name == "cluster_rmsprop":
            emb_params, other_params = [], []

            for name, p in self.model.named_parameters():
                if ('embedding_user' in name) or ('embedding_item' in name):
                    emb_params.append(p)
                else:
                    other_params.append(p)

            param_groups = []

            if emb_params:
                param_groups.append({
                    "params": emb_params,
                    "lr": self.lr,
                    "clustered": True,
                    "alpha_cluster": config.get("alpha", 0.5),
                    "num_clusters": config.get("num_clusters", 16),
                    "recluster_interval": config.get("recluster_interval", 100),
                })

            if other_params:
                param_groups.append({
                    "params": other_params,
                    "lr": self.lr,
                    "clustered": False,
                })

            self.opt = ClusterCoupledRMSProp(param_groups)
        else:
            # 기본: 원래 코드와 같은 Adam
            self.opt = optim.Adam(
                recmodel.parameters(),
                lr=self.lr
            )

    def stageOne(self, users, pos, neg):
        """
        한 step BPR 업데이트:
        - 모델에서 BPR loss와 L2 reg_loss 받아옴
        - reg_loss * weight_decay 더해서 최종 loss 만들고
        - 선택된 옵티마이저(self.opt)로 한 번 업데이트
        """
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss * self.weight_decay
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()

def UniformSample_original(dataset, neg_ratio = 1):
    dataset : BasicDataset
    allPos = dataset.allPos
    start = time()
    if sample_ext:
        S = sampling.sample_negative(dataset.n_users, dataset.m_items,
                                     dataset.trainDataSize, allPos, neg_ratio)
    else:
        S = UniformSample_original_python(dataset)
    return S

def UniformSample_original_python(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    total_start = time()
    dataset : BasicDataset
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        end = time()
        sample_time1 += end - start
    total = time() - total_start
    return np.array(S)

# ===================end samplers==========================
# =====================utils====================================

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def getFileName():
    if world.model_name == 'mf':
        file = f"mf-{world.dataset}-{world.config['latent_dim_rec']}.pth.tar"
    elif world.model_name == 'lgn':
        file = f"lgn-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}.pth.tar"
    return os.path.join(world.FILE_PATH,file)

def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                #TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)


# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1./np.arange(1, k+1))
    pred_data = pred_data/scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)

def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def AUC(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    dataset : BasicDataset
    r_all = np.zeros((dataset.m_items, ))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

# ====================Additional Metrics=============================

def HitRatio_ATk(ground_true, ranked, k):
    return 1.0 if np.sum(ranked[:k]) > 0 else 0.0

# ====================end Metrics=============================
# =========================================================


class MetricsRecorder:
    """
    Simple CSV logger that stores evaluation metrics for every epoch.
    """
    def __init__(self, log_path, topks):
        self.log_path = log_path
        self.topks = list(topks)
        directory = os.path.dirname(log_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.header = self._build_header()
        self._initialized = os.path.exists(log_path) and os.path.getsize(log_path) > 0

    def _build_header(self):
        columns = ['epoch', 'loss', 'precision', 'recall', 'ndcg', 'hr']
        metric_names = ['precision', 'recall', 'ndcg', 'hr']
        for metric in metric_names:
            for k in self.topks:
                columns.append(f"{metric}@{k}")
        columns.extend(['convergence_speed', 'variance'])
        return columns

    def log(self, epoch, metrics, train_loss=None):
        metrics = metrics or {}
        row = {key: 0.0 for key in self.header}
        row['epoch'] = epoch
        if train_loss is not None:
            row['loss'] = float(train_loss)
        for metric_name in ['precision', 'recall', 'ndcg', 'hr']:
            values = metrics.get(metric_name)
            if values is None:
                continue
            if len(values) > 0:
                row[metric_name] = float(values[0])
            for idx, k in enumerate(self.topks):
                column = f"{metric_name}@{k}"
                value = values[idx] if idx < len(values) else 0.0
                row[column] = float(value)
        row['convergence_speed'] = float(metrics.get('convergence_speed', 0.0))
        row['variance'] = float(metrics.get('variance', 0.0))
        with open(self.log_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.header)
            if not self._initialized:
                writer.writeheader()
                self._initialized = True
            writer.writerow(row)
