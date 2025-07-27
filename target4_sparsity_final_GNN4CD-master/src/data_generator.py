import os

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch import optim
import torch.nn.functional as F
from load import get_P, get_Pd, get_W_lg


class Generator(object):
    def __init__(self, N_train=50, N_test=100, generative_model='SBM_multiclass', p_SBM=0.8, q_SBM=0.2, n_classes=2, path_dataset='', num_examples_train=100, num_examples_test=10):
        self.N_train = N_train
        self.N_test = N_test
        self.generative_model = generative_model
        self.p_SBM = p_SBM
        self.q_SBM = q_SBM
        self.n_classes = n_classes
        self.path_dataset = path_dataset
        self.data_train = None
        self.data_test = None
        self.num_examples_train = num_examples_train
        self.num_examples_test = num_examples_test

    def SBM(self, p, q, N):
        W = np.zeros((N, N))

        p_prime = 1 - np.sqrt(1 - p)
        q_prime = 1 - np.sqrt(1 - q)

        n = N // 2

        W[:n, :n] = np.random.binomial(1, p, (n, n))
        W[n:, n:] = np.random.binomial(1, p, (N-n, N-n))
        W[:n, n:] = np.random.binomial(1, q, (n, N-n))
        W[n:, :n] = np.random.binomial(1, q, (N-n, n))
        W = W * (np.ones(N) - np.eye(N))
        W = np.maximum(W, W.transpose())

        perm = torch.randperm(N).numpy()
        blockA = perm < n
        labels = blockA * 2 - 1

        W_permed = W[perm]
        W_permed = W_permed[:, perm]
        return W_permed, labels


    def SBM_multiclass(self, p, q, N, n_classes):

        p_prime = p
        q_prime = q

        prob_mat = np.ones((N, N)) * q_prime

        n = N // n_classes  # 基础类别大小
        remainder = N % n_classes  # 不能整除的剩余部分
        n_last = n + remainder  # 最后一类的大小

        # 先对整除部分进行块状分配
        for i in range(n_classes - 1):  # 处理前 n_classes-1 类
            prob_mat[i * n: (i + 1) * n, i * n: (i + 1) * n] = p_prime

        # 处理最后一类
        start_idx = (n_classes - 1) * n  # 最后一类的起始索引
        prob_mat[start_idx: start_idx + n_last, start_idx: start_idx + n_last] = p_prime

        # 生成邻接矩阵
        W = np.random.rand(N, N) < prob_mat
        W = W.astype(int)

        W = W * (np.ones(N) - np.eye(N))  # 移除自环
        W = np.maximum(W, W.transpose())  # 确保无向图

        # 随机打乱节点顺序
        perm = torch.randperm(N).numpy()

        # 生成类别标签
        labels =np.minimum((perm // n) , n_classes - 1)

        W_permed = W[perm]
        W_permed = W_permed[:, perm]

        #计算P矩阵的特征向量
        prob_mat_permed = prob_mat[perm][:, perm]
        # np.fill_diagonal(prob_mat_permed, 0)  # 去除自环

        eigvals, eigvecs = np.linalg.eigh(prob_mat_permed)
        idx = np.argsort(eigvals)[::-1]
        eigvecs_top = eigvecs[:, idx[:n_classes]]

        return W_permed, labels, eigvecs_top  # 返回前n_classes特征向量

    def create_dataset(self, directory, is_training):
        if (self.generative_model == 'SBM_multiclass'):
            if not os.path.exists(directory):
                os.mkdir(directory)
            if is_training:
                graph_size = self.N_train
                num_graphs = self.num_examples_train
            else:
                graph_size = self.N_test
                num_graphs = self.num_examples_test
            dataset = []
            for i in range(num_graphs):
                W, labels = self.SBM_multiclass(self.p_SBM, self.q_SBM, graph_size, self.n_classes)
                Pm = get_Pm(W)
                Pd = get_Pd(W)
                NB = get_W_lg(W)
                example = {}
                example['W'] = W
                example['labels'] = labels
                dataset.append(example)
            if is_training:
                print ('Saving the training dataset')
            else:
                print ('Saving the testing dataset')
            np.save(directory + '/dataset.npy', dataset)
            if is_training:
                self.data_train = dataset
            else:
                self.data_test = dataset
        else:
            raise ValueError('Generative model {} not supported'.format(self.generative_model))


    def prepare_data(self):
        train_directory = self.generative_model + '_nc' + str(self.n_classes) + '_p' + str(self.p_SBM) + '_q' + str(self.q_SBM) + '_gstr' + str(self.N_train) + '_numtr' + str(self.num_examples_train)
        
        train_path = os.path.join(self.path_dataset, train_directory)
        if os.path.exists(train_path + '/dataset.npy'):
            print('Reading training dataset at {}'.format(train_path))
            self.data_train = np.load(train_path + '/dataset.npy')
        else:
            print('Creating training dataset.')
            self.create_dataset(train_path, is_training=True)
        # load test dataset
        test_directory = self.generative_model + '_nc' + str(self.n_classes) + '_p' + str(self.p_SBM) + '_q' + str(self.q_SBM) + '_gste' + str(self.N_test) + '_numte' + str(self.num_examples_test)
        test_path = os.path.join(self.path_dataset, test_directory)
        if os.path.exists(test_path + '/dataset.npy'):
            print('Reading testing dataset at {}'.format(test_path))
            self.data_test = np.load(test_path + '/dataset.npy')
        else:
            print('Creating testing dataset.')
            self.create_dataset(test_path, is_training=False)


    def sample_single(self, i, is_training=True):
        if is_training:
            dataset = self.data_train
        else:
            dataset = self.data_test
        example = dataset[i]
        if (self.generative_model == 'SBM_multiclass'):
            W_np = example['W']
            labels = np.expand_dims(example['labels'], 0)
            labels_var = torch.from_numpy(labels)
            if is_training:
                labels_var.requires_grad = True
            return W_np, labels_var 


    def sample_otf_single(self, is_training=True, cuda=True):
        if is_training:
            N = self.N_train
        else:
            N = self.N_test
        if self.generative_model == 'SBM':
            W, labels = self.SBM(self.p_SBM, self.q_SBM, N)
        elif self.generative_model == 'SBM_multiclass':
            W, labels,eigvecs_top = self.SBM_multiclass(self.p_SBM, self.q_SBM, N, self.n_classes)
        else:
            raise ValueError('Generative model {} not supported'.format(self.generative_model))

        labels = np.expand_dims(labels, 0)
        labels = torch.from_numpy(labels)
        W = np.expand_dims(W, 0)
        # W = torch.tensor(W, dtype=torch.float32)  # 不加 requires_grad

        return W, labels, eigvecs_top

