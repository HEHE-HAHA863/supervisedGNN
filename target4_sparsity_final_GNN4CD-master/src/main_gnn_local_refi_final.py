import os
import openpyxl
from data_generator import Generator
from load import get_gnn_inputs
from models import GNN_multiclass, GNN_multiclass_second_period
import time
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch import Tensor
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from get_infor_from_first_step import get_second_period_labels_single, test_first_get_second_period_labels_single, in_sample_test_first_get_second_period_labels_single
from losses import compute_loss_multiclass, compute_accuracy_multiclass, compute_accuracy_spectural
from load_local_refinement import get_gnn_inputs_local_refinement
import sys
import os
import time
import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
from sklearn.cluster import KMeans


def setup_logger(prefix="main_gnn"):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    log_filename = f"{prefix}_{timestamp}_{pid}.log"
    logfile = open(log_filename, "w", buffering=1)  # 行缓冲
    sys.stdout = logfile
    sys.stderr = logfile
    print(f"[Logger initialized] Logging to: {log_filename}")


parser = argparse.ArgumentParser()

###############################################################################
#                             General Settings                                #
#                          提前配置参数，方便后面使用                              #
###############################################################################

parser.add_argument('--num_examples_train', nargs='?', const=1, type=int,
                    default=int(6000))
parser.add_argument('--num_examples_test', nargs='?', const=1, type=int,
                    default=int(100))
parser.add_argument('--edge_density', nargs='?', const=1, type=float,
                    default=0.2)
parser.add_argument('--p_SBM', nargs='?', const=1, type=float,
                    default=0.2)
parser.add_argument('--q_SBM', nargs='?', const=1, type=float,
                    default=0.1)
parser.add_argument('--random_noise', action='store_true')
parser.add_argument('--noise', nargs='?', const=1, type=float, default=2)
parser.add_argument('--noise_model', nargs='?', const=1, type=int, default=2)
#########################
#parser.add_argument('--generative_model', nargs='?', const=1, type=str,
#                    default='ErdosRenyi')
parser.add_argument('--generative_model', nargs='?', const=1, type=str,
                    default='SBM_multiclass')
parser.add_argument('--batch_size', nargs='?', const=1, type=int, default=1)
parser.add_argument('--mode', nargs='?', const=1, type=str, default='train')
default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
parser.add_argument('--path_gnn', nargs='?', const=1, type=str, default=default_path)
parser.add_argument('--path_local_refinement', nargs='?', const=1, type=str, default='')

parser.add_argument('--filename_existing_gnn', nargs='?', const=1, type=str, default='')
parser.add_argument('--filename_existing_gnn_local_refinement', nargs='?', const=1, type=str, default='')

parser.add_argument('--print_freq', nargs='?', const=1, type=int, default=100)
parser.add_argument('--test_freq', nargs='?', const=1, type=int, default=500)
parser.add_argument('--save_freq', nargs='?', const=1, type=int, default=2000)
parser.add_argument('--clip_grad_norm', nargs='?', const=1, type=float,
                    default=40.0)
parser.add_argument('--freeze_bn', dest='eval_vs_train', action='store_true')
parser.set_defaults(eval_vs_train=True)

###############################################################################
#                                 GNN Settings                                #
###############################################################################

parser.add_argument('--num_features', nargs='?', const=1, type=int,
                    default=8)
parser.add_argument('--num_layers', nargs='?', const=1, type=int,
                    default=50)
parser.add_argument('--n_classes', nargs='?', const=1, type=int,
                    default=3)
parser.add_argument('--J', nargs='?', const=1, type=int, default=4)
parser.add_argument('--N_train', nargs='?', const=1, type=int, default=1000)
parser.add_argument('--N_test', nargs='?', const=1, type=int, default=1000)
parser.add_argument('--lr', nargs='?', const=1, type=float, default=1e-3)

args = parser.parse_args()

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
    # torch.cuda.manual_seed(0)
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor
    # torch.manual_seed(1)

batch_size = args.batch_size
criterion = nn.CrossEntropyLoss()
template1 = '{:<10} {:<10} {:<10} {:<15} {:<10} {:<10} {:<10}'
template2 = '{:<10} {:<10.5f} {:<10.5f} {:<15} {:<10} {:<10} {:<10.3f} \n'
template3 = '{:<10} {:<10} {:<10} '
template4 = '{:<10} {:<10.5f} {:<10.5f} \n'

import torch


#########################################################

def spectral_clustering_adj(A: np.ndarray, k: int, true_labels, normalized: bool = True):
    # 计算拉普拉斯
    A = A.squeeze(0)  # 从 (1, N, N) 得到 (N, N)

    if normalized:
        L = laplacian(A, normed=True)
    else:
        L = laplacian(A, normed=False)

    # 特征值分解（提取 k 个最小特征向量）
    eigvals, eigvecs = eigh(L, subset_by_index=(0, k - 1))

    # 对向量行归一化
    U = eigvecs
    U_norm = U / np.linalg.norm(U, axis=1, keepdims=True)

    # KMeans 聚类
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = kmeans.fit_predict(U_norm)
    acc_spectrual_clustering, best_matched_pred = compute_accuracy_spectural(labels, true_labels, k)

    return acc_spectrual_clustering

cached_graphs = []
cached_labels = []

##Define the train function we need to train the first-period GNN function
def train_single_first_period(gnn, optimizer, gen, n_classes, iter):
    # --- 1. 生成图 ---
    W, labels, eigvecs_top = gen.sample_otf_single(is_training=True, cuda=torch.cuda.is_available())
    # --- 2. 缓存图和标签到 CPU 列表（保留 batch 维度）---
    W_cpu = W
    labels_cpu = labels

    cached_graphs.append(W_cpu)       # 每个元素都是 (1, 1000, 1000)
    cached_labels.append(labels_cpu)  # 每个元素都是 (1, 1000)

    labels = labels.type(dtype_l).to(device)  # (1, N)

    # --- 3. 正常训练流程 ---
    start = time.time()
    WW, x = get_gnn_inputs(W, args.J)
    WW = WW.to(device)
    x = x.to(device)

    optimizer.zero_grad(set_to_none=True)
    pred = gnn(WW.type(dtype), x.type(dtype))
    loss = compute_loss_multiclass(pred, labels, n_classes)

    loss.backward()
    nn.utils.clip_grad_norm_(gnn.parameters(), args.clip_grad_norm)
    optimizer.step()

    acc, best_matched_pred = compute_accuracy_multiclass(pred, labels, n_classes)
    loss_value = loss.item()
    elapsed_time = time.time() - start

    info = ['iter', 'avg loss', 'avg acc', 'edge_density', 'noise', 'model', 'elapsed']
    out = [iter, loss_value, acc, args.edge_density, args.noise, 'GNN', elapsed_time]

    print(template1.format(*info))
    print(template2.format(*out))

    # 释放显存
    WW = None
    x = None
    torch.cuda.empty_cache()

    return loss_value, acc


def train_first_period(gnn, gen, n_classes=args.n_classes, iters=args.num_examples_train):
    gnn.train()
    optimizer = torch.optim.Adamax(gnn.parameters(), lr=args.lr)
    loss_lst = np.zeros([iters])
    acc_lst = np.zeros([iters])

    for it in range(iters):
        loss_single, acc_single = train_single_first_period(gnn, optimizer, gen, n_classes, it)
        loss_lst[it] = loss_single
        acc_lst[it] = acc_single
        torch.cuda.empty_cache()


def get_available_device():
    for i in range(torch.cuda.device_count()):
        try:
            # 尝试分配临时张量测试显存
            torch.cuda.set_device(i)
            torch.zeros(1).cuda()
            return torch.device(f"cuda:{i}")
        except RuntimeError:
            continue
    return torch.device("cpu")


device = get_available_device()

##############Get the labels from the first period and use this labels to train the local refinement###################
def train_single_local_refinement(gnn_first_period, gnn_local_refine, n_classes, W, true_labels ,optimizer, iter):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sam_com_matrix, pred_labels = get_second_period_labels_single(gnn_first_period, W, true_labels,n_classes, args)

    true_labels = true_labels.type(dtype_l)

    start = time.time()
    WW, x = get_gnn_inputs_local_refinement(W, args.J, sam_com_matrix, pred_labels, n_classes)

    # **移动数据到计算设备 (GPU/CPU)**
    WW = WW.to(device)
    x = x.to(device)

    optimizer.zero_grad(set_to_none=True)
    pred = gnn_local_refine(WW.type(dtype), x.type(dtype))

    loss = compute_loss_multiclass(pred, true_labels, n_classes)  # 计算损失
    loss.backward()
    nn.utils.clip_grad_norm_(gnn_local_refine.parameters(), args.clip_grad_norm)
    optimizer.step()

    acc, best_matched_pred = compute_accuracy_multiclass(pred, true_labels, n_classes)

    elapsed_time = time.time() - start

    if torch.cuda.is_available():
        loss_value = float(loss.data.cpu().numpy())
    else:
        loss_value = float(loss.data.numpy())

    info = ['iter', 'avg loss', 'avg acc', 'edge_density',
            'noise', 'model', 'elapsed']
    out = [iter, loss_value, acc, args.edge_density,
           args.noise, 'GNN', elapsed_time]

    print(template1.format(*info))
    print(template2.format(*out))

    # **释放 GPU 显存**
    WW = None
    x = None

    return loss_value, acc


def train_local_refinement(gnn_first_period, gnn_local_refine, n_classes=args.n_classes, iters=args.num_examples_train):
    gnn_local_refine.train()
    optimizer = torch.optim.Adamax(gnn_local_refine.parameters(), lr=args.lr)

    loss_lst = np.zeros([iters])
    acc_lst = np.zeros([iters])

    for it in range(iters):
        W_i = cached_graphs[it]
        true_labels_i = cached_labels[it]

        loss_single, acc_single = train_single_local_refinement(gnn_first_period, gnn_local_refine, n_classes, W_i, true_labels_i,
                                                                optimizer, it)

        loss_lst[it] = loss_single
        acc_lst[it] = acc_single

        torch.cuda.empty_cache()

def get_insample_acc_lost_single(gnn_first_period, gnn_local_refine, W_i, labels_i, n_classes, args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    W, true_labels = W_i, labels_i

    sam_com_matrix, pred_label_first ,first_in_sample_loss, first_in_sample_acc \
     = in_sample_test_first_get_second_period_labels_single(gnn_first_period, W, true_labels,n_classes, args)

    WW, x = get_gnn_inputs_local_refinement(W, args.J, sam_com_matrix, pred_label_first, n_classes)

    if torch.cuda.is_available():
        WW = WW.to(device)
        x = x.to(device)

    pred_single_second = gnn_local_refine(WW.type(dtype), x.type(dtype))

    # 计算第二阶段损失和准确率
    second_in_sample_loss = compute_loss_multiclass(pred_single_second, true_labels, n_classes)
    second_in_sample_acc, best_matched_pred = compute_accuracy_multiclass(pred_single_second, true_labels, n_classes)

    WW = None
    x = None

    return first_in_sample_loss, first_in_sample_acc, float(second_in_sample_loss.data.cpu().numpy()), second_in_sample_acc

def get_insample_acc_lost(gnn_first_period, gnn_local_refinement, n_classes, iters=args.num_examples_test,
                          filename="in_sample_test_results_sparsity.csv"):

    gnn_first_period.train()
    gnn_local_refinement.train()

    # in_sample_loss_lst_first = np.zeros([iters])
    in_sample_acc_lst_first = np.zeros([iters])

    # in_sample_loss_lst_second = np.zeros([iters])
    in_sample_acc_lst_second = np.zeros([iters])

    for it in range(iters):
        W_i = cached_graphs[it]
        labels_i = cached_labels[it]

        first_in_sample_loss, first_in_sample_acc, second_in_sample_loss, second_in_sample_acc = get_insample_acc_lost_single(
            gnn_first_period,
            gnn_local_refinement, W_i, labels_i, n_classes, args)

        # in_sample_loss_lst_first[it] = first_in_sample_loss
        in_sample_acc_lst_first[it] = first_in_sample_acc

        # in_sample_loss_lst_second[it] = second_in_sample_loss
        in_sample_acc_lst_second[it] = second_in_sample_acc

        torch.cuda.empty_cache()
    # 计算均值和标准差
    first_avg_test_acc = np.mean(in_sample_acc_lst_first)
    first_std_test_acc = np.std(in_sample_acc_lst_first)

    second_avg_test_acc = np.mean(in_sample_acc_lst_second)
    second_std_test_acc = np.std(in_sample_acc_lst_second)

    n = args.N_train  # 或者 N_test，也可以统一都用 N
    logn_div_n = np.log(n) / n

    a = args.p_SBM / logn_div_n
    b = args.q_SBM / logn_div_n
    k = args.n_classes

    snr = (a - b) ** 2 / (k * (a + (k - 1) * b))

    df = pd.DataFrame([{
        "n_classes": args.n_classes,
        "p_SBM": args.p_SBM,
        "q_SBM": args.q_SBM,
        "J": args.J,
        "N_train": args.N_train,
        "N_test": args.N_test,
        "first_avg_test_acc": first_avg_test_acc,
        "first_std_test_acc": first_std_test_acc,
        "second_avg_test_acc": second_avg_test_acc,
        "second_std_test_acc": second_std_test_acc,
        "SNR": snr
    }])

    # 追加模式写入文件，防止覆盖
    df.to_csv(filename, mode='a', index=False, header=not pd.io.common.file_exists(filename))


def test_single_local_refinement(gnn_first_period, gnn_local_refinement, gen, n_classes, args, iter):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test the first period and get the labels we need to do local refinement
    sam_com_matrix, true_labels, pred_label_first, W, loss_test_first, acc_test_first, eigvecs_top \
        = test_first_get_second_period_labels_single(gnn_first_period, gen, n_classes, args)

    # 计算谱聚类的正确率以此来用于对比
    acc_spectrual_clustering = spectral_clustering_adj(W, n_classes, true_labels)

    # 获取GNN输入
    WW, x = get_gnn_inputs_local_refinement(W, args.J, sam_com_matrix, pred_label_first, n_classes)

    if torch.cuda.is_available():
        WW = WW.to(device)
        x = x.to(device)

    pred_single_second = gnn_local_refinement(WW.type(dtype), x.type(dtype))

    # final_features = pred_single_second.detach().cpu().numpy().squeeze(0)
    # final_features /= np.linalg.norm(final_features, axis=0, keepdims=True)

    penultimate_features = gnn_local_refinement.get_penultimate_output().detach().cpu().numpy().squeeze(0)
    penultimate_features /= np.linalg.norm(penultimate_features, axis=0, keepdims=True)

    # 计算第二阶段损失和准确率
    loss_test_second = compute_loss_multiclass(pred_single_second, true_labels, n_classes)
    acc_test_second, best_matched_pred = compute_accuracy_multiclass(pred_single_second, true_labels, n_classes)

    pred_label_second = best_matched_pred

    # 获取样本数（节点数）
    N = true_labels.shape[1]

    # 构造 Excel 数据内容
    data = {
        'True_Label': true_labels.squeeze(0).cpu().numpy(),
        'Pred_Label_First': pred_label_first.reshape(-1),
        'Pred_Label_Second': pred_label_second.reshape(-1),
        'Loss_First': [float(loss_test_first)] * N,
        'Acc_First': [float(acc_test_first)] * N,
        'Loss_Second': [float(loss_test_second.data.cpu().numpy())] * N,
        'Acc_Second': [float(acc_test_second)] * N,
    }

    # 添加 GNN 特征列
    for i in range(args.num_features):
        data[f'penultimate_GNN_Feature{i + 1}'] = penultimate_features[:, i]
    # for i in range(n_classes):
    #     data[f'final_GNN_Feature{i + 1}'] = final_features[:, i]
    for i in range(n_classes):
        data[f'eigvecs_top{i + 1}'] = eigvecs_top[:, i]

    df = pd.DataFrame(data)

    # 1. 创建总目录（只建一次）
    root_folder = "penultimate_GNN_Feature"
    os.makedirs(root_folder, exist_ok=True)

    # 2. 创建子目录（按 n_classes 分类）
    subfolder_name = f"penultimate_GNN_Feature_nclass_{n_classes}"
    subfolder_path = os.path.join(root_folder, subfolder_name)
    os.makedirs(subfolder_path, exist_ok=True)

    # 3. 构建完整输出路径
    output_filename = (
        f"local_refine_gnn_classes={n_classes}_p={args.p_SBM}_q={args.q_SBM}"
        f"_j={args.J}_nlyr={args.num_layers}.xlsx"
    )
    output_path = os.path.join(subfolder_path, output_filename)

    # 3. iter < 10 才写入 Excel
    if iter < 10:
        if iter == 0:
            df.to_excel(output_path, sheet_name=f'Iteration_{iter}', index=False)
        else:
            with pd.ExcelWriter(output_path, mode='a', engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=f'Iteration_{iter}', index=False)

    # 清理内存
    WW = None
    x = None

    return loss_test_first, acc_test_first, float(loss_test_second.data.cpu().numpy()), acc_test_second, acc_spectrual_clustering


def test_local_refinement(gnn_first_period, gnn_local_refinement, n_classes, gen, iters=args.num_examples_test,
                          filename="out_sample_test_results_sparsity.csv"):
    gnn_first_period.train()
    gnn_local_refinement.train()

    loss_lst_first = np.zeros([iters])
    acc_lst_first = np.zeros([iters])

    loss_lst_second = np.zeros([iters])
    acc_lst_second = np.zeros([iters])

    acc_spectrual_clustering_list = np.zeros([iters])

    for it in range(iters):
        loss_test_first, acc_test_first, loss_test_second, acc_test_second, acc_spectrual_clustering = test_single_local_refinement(
            gnn_first_period,
            gnn_local_refinement, gen, n_classes, args, it)

        loss_lst_first[it] = loss_test_first
        acc_lst_first[it] = acc_test_first

        loss_lst_second[it] = loss_test_second
        acc_lst_second[it] = acc_test_second

        acc_spectrual_clustering_list[it] = acc_spectrual_clustering

        torch.cuda.empty_cache()
    # 计算均值和标准差
    first_avg_test_acc = np.mean(acc_lst_first)
    first_std_test_acc = np.std(acc_lst_first)

    second_avg_test_acc = np.mean(acc_lst_second)
    second_std_test_acc = np.std(acc_lst_second)

    spectrual_clustering_avg_test_acc = np.mean(acc_spectrual_clustering_list)
    spectrual_clustering_std_test_acc = np.std(acc_spectrual_clustering_list)

    n = args.N_train  # 或者 N_test，也可以统一都用 N
    logn_div_n = np.log(n) / n

    a = args.p_SBM / logn_div_n
    b = args.q_SBM / logn_div_n
    k = args.n_classes

    snr = (a - b) ** 2 / (k * (a + (k - 1) * b))

    df = pd.DataFrame([{
        "n_classes": args.n_classes,
        "p_SBM": args.p_SBM,
        "q_SBM": args.q_SBM,
        "J": args.J,
        "N_train": args.N_train,
        "N_test": args.N_test,
        "first_avg_test_acc": first_avg_test_acc,
        "first_std_test_acc": first_std_test_acc,
        "second_avg_test_acc": second_avg_test_acc,
        "second_std_test_acc": second_std_test_acc,
        "spectrual_clustering_avg_test_acc": spectrual_clustering_avg_test_acc,
        "spectrual_clustering_std_test_acc": spectrual_clustering_std_test_acc,
        "SNR": snr
    }])

    # 追加模式写入文件，防止覆盖
    df.to_csv(filename, mode='a', index=False, header=not pd.io.common.file_exists(filename))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    try:
        setup_logger("run_gnn")
        gen = Generator()
        gen.N_train = args.N_train
        gen.N_test = args.N_test
        gen.edge_density = args.edge_density
        gen.p_SBM = args.p_SBM
        gen.q_SBM = args.q_SBM
        gen.random_noise = args.random_noise
        gen.noise = args.noise
        gen.noise_model = args.noise_model
        gen.generative_model = args.generative_model
        gen.n_classes = args.n_classes

        # 1. 创建总模型文件夹
        root_model_dir = "model_GNN"
        os.makedirs(root_model_dir, exist_ok=True)

        # 2. 创建子目录（按 n_classes 分类）
        folder_name = f"GNN_model_first&local_nclass_{args.n_classes}"
        full_save_dir = os.path.join(root_model_dir, folder_name)
        os.makedirs(full_save_dir, exist_ok=True)

        # 3. 构造保存路径
        filename_first = (
            f'gnn_J{args.J}_lyr{args.num_layers}_classes{args.n_classes}_p={args.p_SBM}_q={args.q_SBM}'
        )
        path_first = os.path.join(full_save_dir, filename_first)

        filename_second = (
            f'local+refin_gnn_J{args.J}_lyr{args.num_layers}_classes{args.n_classes}_p={args.p_SBM}_q={args.q_SBM}'
        )
        path_second = os.path.join(full_save_dir, filename_second)


        if args.mode == "train":
            print(
                f"[阶段一] 训练 GNN：n_classes={args.n_classes}, p={args.p_SBM}, q={args.q_SBM}, layers={args.num_layers}, J = {args.J}")
            # 初始化并训练第一阶段
            if args.generative_model == 'SBM_multiclass':
                gnn_first_period = GNN_multiclass(args.num_features, args.num_layers, args.J + 3,
                                                  n_classes=args.n_classes)
            if torch.cuda.is_available():
                gnn_first_period = gnn_first_period.to(device)

            train_first_period(gnn_first_period, gen, args.n_classes)

            print(f'Saving first-period GNN to {path_first}')
            torch.save(gnn_first_period.cpu(), path_first)
            if torch.cuda.is_available():
                gnn_first_period = gnn_first_period.to('cuda')

            print(
                f"[阶段二] Local Refinement 开始：n_classes={args.n_classes}, p={args.p_SBM}, q={args.q_SBM}, layers={args.num_layers}, J = {args.J}")
            if args.generative_model == 'SBM_multiclass':
                gnn_local_refinement = GNN_multiclass_second_period(args.num_features, args.num_layers, args.J + 4,
                                                                    n_classes=args.n_classes)
            if torch.cuda.is_available():
                gnn_local_refinement = gnn_local_refinement.to(device)

            train_local_refinement(
                gnn_first_period,
                gnn_local_refinement,
                n_classes=args.n_classes,
                iters=args.num_examples_train
            )

            print(f'Saving local refinement GNN to {path_second}')
            torch.save(gnn_local_refinement.cpu(), path_second)
            if torch.cuda.is_available():
                gnn_local_refinement = gnn_local_refinement.to('cuda')

            get_insample_acc_lost(gnn_first_period, gnn_local_refinement,n_classes=args.n_classes )

        else:  # ======================== test 模式 ========================
            print("[TEST 模式] 直接加载已训练好的模型...")
            gnn_first_period = torch.load(path_first, map_location=device)
            gnn_local_refinement = torch.load(path_second, map_location=device)
            if torch.cuda.is_available():
                gnn_first_period = gnn_first_period.to(device)
                gnn_local_refinement = gnn_local_refinement.to(device)

        # ============================== 测试阶段 ===================================
        print("[测试阶段] 开始...")
        test_local_refinement(
            gnn_first_period,
            gnn_local_refinement,
            args.n_classes,
            gen,
            iters=args.num_examples_test
        )

    except Exception as e:
        import traceback
        traceback.print_exc()

