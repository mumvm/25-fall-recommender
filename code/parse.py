'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--bpr_batch', type=int,default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int,default=3,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int,default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float,default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='gowalla',
                        help="available datasets: [lastfm, gowalla, yelp2018, amazon-book]")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[20]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int,default=1,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str,default="lgn")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--epochs', type=int,default=5)
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'adam', 'sgd', 'rmsprop', 'lamb', 'cluster'],
                        help="optimizer for BPR training")
    # ------------------ cluster option ------------------
    parser.add_argument('--cluster_alpha', type=float, default=0.3,
                        help="gradient mixing strength for cluster-coupled AdamW")
    parser.add_argument('--cluster_k', type=int, default=16,
                        help="number of clusters for embedding rows")
    parser.add_argument('--cluster_interval', type=int, default=200,
                        help="steps between reclustering runs")
    parser.add_argument('--cluster_warmup', type=int, default=10,
                        help="skip clustering during the first N optimizer steps")
    parser.add_argument('--cluster_min_rows', type=int, default=2,
                        help="minimum rows needed to apply clustering")
    parser.add_argument('--cluster_eps', type=float, default=1e-8,
                        help="epsilon value for cluster-coupled AdamW")
    # ---- aliases for notebook commands ----
    parser.add_argument('--alpha', type=float, default=None,
                        help="alias of cluster_alpha (for shared notebooks)")
    parser.add_argument('--num_clusters', type=int, default=None,
                        help="alias of cluster_k (for shared notebooks)")
    parser.add_argument('--recluster_interval', type=int, default=None,
                        help="alias of cluster_interval (for shared notebooks)")
    # --------------------- end ---------------------
    parser.add_argument('--test_interval', type=int, default=1,
                        help='run evaluation every N epochs')
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='lgn', help='rec-model, support [mf, lgn]')
    args = parser.parse_args()
    # alias handling to keep compatibility with shared notebooks
    if args.alpha is not None:
        args.cluster_alpha = args.alpha
    if args.num_clusters is not None:
        args.cluster_k = args.num_clusters
    if args.recluster_interval is not None:
        args.cluster_interval = args.recluster_interval
    return args
