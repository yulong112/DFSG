import numpy as np
import scipy.sparse as sp
import torch
# from torch_geometric.datasets import Planetoid
# from torch_geometric.transforms import NormalizeFeatures, GDC
import scipy.io as scio
import torch
from scipy.sparse import coo_matrix
import scipy.sparse.linalg as spla
from SPG_toolbox import FSPG_graph_construction_bigdata251103
from scipy.sparse import spdiags, csr_matrix, diags
from sklearn.preprocessing import MinMaxScaler
import spectral as spy
import matplotlib.pyplot as plt


# def encode_onehot(labels):
#     n_clz = torch.unique(labels).size(0)
#     source = torch.ones((labels.shape[0], 1), dtype=torch.float32)
#     labels_onehot = torch.zeros((labels.shape[0], n_clz), dtype=torch.float32)
#     labels_onehot = labels_onehot.scatter_(dim=1, index=labels.unsqueeze(1), src=source)
#     return labels_onehot

# def encode_onehotV2(labels):
#     n_clz = torch.max(labels)+1
#     labels_onehot = torch.zeros((labels.shape[0], n_clz), dtype=torch.float32)
#     # 找到有效的标签（不等于 -1）
#     valid_mask = labels >= 0
#     valid_labels = labels[valid_mask]
#     # 将有效标签 one-hot 编码
#     source = torch.ones((valid_labels.shape[0], 1), dtype=torch.float32)
#     labels_onehot[valid_mask] = labels_onehot[valid_mask].scatter_(dim=1, index=valid_labels.unsqueeze(1), src=source)
#     return labels_onehot


# class Data_info():
#     x = None
#     edge_index=None
#     num_nodes=None
#     num_edges=None


# def k_nearest_neighbors_sparsify(A, k):
#     N = A.shape[0]
#     # 初始化一个全为零的稀疏矩阵
#     A_sparse = np.zeros_like(A)
#
#     # 遍历每一行，找到每一行的前 k 大的元素索引
#     for i in range(N):
#         # row = A[i, :]
#         row = np.array(A[i, :])[0]
#         # 找到第 k 大元素的索引
#         knn_idx = np.argsort(row)[-k:]  # 返回最大的 k 个索引
#
#         # 保留 k 近邻对应的位置
#         A_sparse[i, knn_idx] = A[i, knn_idx]
#
#     # 保证矩阵对称：A_sparse = max(A_sparse, A_sparse^T)
#     A_sparse = np.maximum(A_sparse, A_sparse.T)
#
#     return A_sparse


def SemiSupervised_Graph_Classifier_v2(Laplace_Mat, initial_Labels, TestSamLoc, alpha):
    N = initial_Labels.shape[0]

    # 构建 (1 - alpha) * I + alpha * Laplace_Mat
    UU = (1 - alpha) * sp.eye(N) + alpha * Laplace_Mat

    # 计算分类函数 F = (I - alpha S)^{-1} * Y
    F = sp.linalg.spsolve(UU, initial_Labels)

    F = F.toarray()
    # 获取每行中最大值对应的标签
    PredLabels = np.argmax(F, axis=1)

    # 计算 confidence coefficient
    confidence_coef = F / np.sum(F, axis=1, keepdims=True)

    # 保存所有样本的预测标签
    PredLabels_ALL = PredLabels.copy()

    # 仅返回 TestSamLoc 对应的预测标签
    PredLabels = PredLabels[TestSamLoc]

    return PredLabels, PredLabels_ALL, confidence_coef

# def cal_test_acc(y_test, y_pred):
#     acc = torch.eq(torch.argmax(y_test, dim=1), torch.argmax(y_pred, dim=1))
#     acc = torch.sum(acc) * 1. / y_test.shape[0]
#     return acc

# def normalize_adj(adj):
#     """Symmetrically normalize adjacency matrix."""
#     adj = sp.coo_matrix(adj)
#     rowsum = np.array(adj.sum(1))
#     d_inv_sqrt = np.power(rowsum, -0.5).flatten()
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#     d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
#     return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).transpose().tocoo()

# def preprocess_adj(adj):
#     """Preprocessing of adjacency matrix for simple GCN_Model model and conversion to tuple representation."""
#     adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
#     return adj_normalized


def superpixel_mean(features, superpix_labels, sp_num):
    # features: [N, C]   每个像素的特征或预测
    # superpix_labels: [N]，标签范围为 1~sp_num
    # 返回: [sp_num, C]
    device = features.device
    superpix_labels = superpix_labels.long() - 1  # 转为 0-based

    counts = torch.bincount(superpix_labels, minlength=sp_num).unsqueeze(1).float().to(device)
    sums = torch.zeros((sp_num, features.size(1)), device=device)
    sums.index_add_(0, superpix_labels, features)
    means = sums / counts.clamp(min=1.0)
    return means


def compute_lossV2(predict: torch.Tensor, reallabel_onehot: torch.Tensor, reallabel_mask: torch.Tensor,
                   superpix_img1: torch.Tensor, superpix_img2: torch.Tensor, Lap1: torch.Tensor, Lap2: torch.Tensor,
                   sp_num1: torch.int, sp_num2: torch.int, nowepoch: torch.int):
    real_labels = reallabel_onehot
    we = -torch.mul(real_labels, torch.log(predict))
    we = torch.mul(we, reallabel_mask)
    pool_cross_entropy = torch.sum(we)

    predict_X1 = superpixel_mean(predict, superpix_img1, sp_num1)
    predict_X2 = superpixel_mean(predict, superpix_img2, sp_num2)

    graph_regularizer1 = torch.trace(torch.matmul(predict_X1.T, torch.matmul(Lap1, predict_X1)))
    graph_regularizer2 = torch.trace(torch.matmul(predict_X2.T, torch.matmul(Lap2, predict_X2)))

    if nowepoch < 50:
        # 从 0.5 线性减少到 0.05
        betaloss = (0.5 - 0.05) * (50 - nowepoch) / (50 - 0) + 0.05
    elif nowepoch < 100:
        # 从 0.05 线性减少到 0.005
        betaloss = (0.05 - 0.005) * (100 - nowepoch) / (100 - 50) + 0.005
    elif nowepoch < 150:
        # 从 0.005 线性减少到 0.001
        betaloss = (0.005 - 0.001) * (150 - nowepoch) / (150 - 100) + 0.001
    elif nowepoch < 200:
        # 从 0.001 线性减少到 0.0001
        betaloss = (0.001 - 0.0001) * (200 - nowepoch) / (200 - 150) + 0.0001
    else:
        # 大于 500 时, betaloss = 0.001
        betaloss = 0.00001

    ## Zaoyuan, IndianPines, higher acc
    # betaloss = 0.005

    pool_cross_entropy = pool_cross_entropy + betaloss * (graph_regularizer1 + graph_regularizer2)

    return pool_cross_entropy


def compute_loss(predict: torch.Tensor, reallabel_onehot: torch.Tensor, reallabel_mask: torch.Tensor):
    real_labels = reallabel_onehot
    we = -torch.mul(real_labels, torch.log(predict))
    we = torch.mul(we, reallabel_mask)
    pool_cross_entropy = torch.sum(we)
    return pool_cross_entropy


def Draw_Classification_Map(label, name: str, scale: float = 4.0, dpi: int = 400):
    '''
    get classification map , then save to given path
    :param label: classification label, 2D
    :param name: saving path and file's name
    :param scale: scale of image. If equals to 1, then saving-size is just the label-size
    :param dpi: default is OK
    :return: null
    '''
    fig, ax = plt.subplots()
    numlabel = np.array(label)
    v = spy.imshow(classes=numlabel.astype(np.int16), fignum=fig.number)
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    foo_fig = plt.gcf()  # 'get current figure'
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)
    pass


def GT_To_One_Hot(gt, class_count):
    '''
    Convet Gt to one-hot labels
    :param gt:
    :param class_count:
    :return:
    '''
    height, width=gt.shape
    GT_One_Hot = []  # 转化为one-hot形式的标签
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            temp = np.zeros(class_count, dtype=np.float32)
            if gt[i, j] != 0:
                temp[int(gt[i, j]) - 1] = 1
            GT_One_Hot.append(temp)
    GT_One_Hot = np.reshape(GT_One_Hot, [height, width, class_count])
    return GT_One_Hot


def extract_validation_samples(test_samples_gt, val_ratio, random_seed=42):
    """
    从 test_samples_gt 中按类别比例抽取验证集样本

    Args:
        test_samples_gt: np.ndarray, shape (H, W), 值为 0~C
        val_ratio: float, 每类抽取比例 (0~1)
        random_seed: int, 随机种子保证可复现

    Returns:
        val_samples_gt: np.ndarray, shape (H, W), 验证集位置保留原标签，其余为0
    """
    np.random.seed(random_seed)
    H, W = test_samples_gt.shape
    val_mask = np.zeros_like(test_samples_gt, dtype=bool)

    # 获取所有类别（忽略背景0）
    unique_labels = np.unique(test_samples_gt)
    unique_labels = unique_labels[unique_labels != 0]

    for label in unique_labels:
        # 找到该类别所有像素的位置
        positions = np.argwhere(test_samples_gt == label)
        num_samples = len(positions)

        # 计算需要选取的验证集数量（至少取1个，如果样本数 > 0）
        num_val = max(1, int(num_samples * val_ratio)) if num_samples > 0 else 0
        num_val = min(num_val, num_samples)  # 不能超过总数

        if num_val > 0:
            # 随机选择索引
            chosen_indices = np.random.choice(num_samples, num_val, replace=False)
            chosen_positions = positions[chosen_indices]
            val_mask[chosen_positions[:, 0], chosen_positions[:, 1]] = True

    # 生成验证集矩阵：验证集位置保留原标签，其余为0
    val_samples_gt = np.where(val_mask, test_samples_gt, 0)

    return val_samples_gt

def calculate_f1_score_simple(network_output, target_onehot, available_label_idx, num_classes, epsilon=1e-7):
    """
    使用混淆矩阵计算F1 Score的简化版本
    """
    pred = torch.argmax(network_output, 1)
    target = torch.argmax(target_onehot, 1)

    # 只考虑有效标签
    valid_mask = (available_label_idx != 0)
    pred_valid = pred[valid_mask]
    target_valid = target[valid_mask]

    # 构建混淆矩阵
    conf_matrix = torch.zeros(num_classes, num_classes, device=network_output.device)
    for t, p in zip(target_valid, pred_valid):
        conf_matrix[t.long(), p.long()] += 1

    # 计算每个类别的TP, FP, FN
    tp = torch.diag(conf_matrix)
    fp = conf_matrix.sum(dim=0) - tp  # 列和减去TP
    fn = conf_matrix.sum(dim=1) - tp  # 行和减去TP

    # 计算F1 Score
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1_per_class = 2 * (precision * recall) / (precision + recall + epsilon)

    f1_mean = f1_per_class.mean()

    return f1_mean, f1_per_class

def A2normLap(WeightMat):
    WeightMat = WeightMat.toarray()
    N = WeightMat.shape[0]
    d = np.array(WeightMat.sum(axis=1)).flatten()  # 计算每行的和 (需要转换为数组)
    # 创建 Dinv 对角矩阵，使用 spdiags
    Dinv = diags(d ** (-0.5), 0)
    # 计算Laplace矩阵
    I = np.eye(N)
    normLap = I - Dinv @ WeightMat @ Dinv
    return normLap

def fmapminmax(HyperCube):
    # 1. 将 HyperCube 重塑为二维数组 (height*width, bands)
    input_regular = HyperCube.reshape(-1, HyperCube.shape[2])
    # 2. 使用 MinMaxScaler 进行归一化
    scaler = MinMaxScaler(feature_range=(0, 255))
    output_regular = scaler.fit_transform(input_regular)
    # 3. 重塑归一化后的数组为原来的形状 (height, width, bands)
    MHyperCube = output_regular.reshape(HyperCube.shape[0], HyperCube.shape[1], HyperCube.shape[2])
    return MHyperCube

def AtoGCN_A(A):
    A = A.toarray()
    N = A.shape[0]
    I = np.eye(N)
    A = A+I
    # 计算每行的和，得到度数向量 d
    d = np.sum(A, axis=1)
    # 确保 d 是一维数组
    # d = np.ravel(d)
    # 构建对角矩阵 Dinv
    Dinv = diags(d**(-0.5), 0)
    # A = np.dot(A, Dinv)
    # A = np.dot(Dinv, A)
    A = Dinv@A@Dinv
    return A

def DFSG_gen(HyperCube, superpix_num0, superpix_allnum, conn_pattern,
             TrainSamLoc, TrainLabels, class_count, dist_mode2, center_mode2):
    WeightMat, superpix_img0, Superpixel_mean_Features = FSPG_graph_construction_bigdata251103(
        HyperCube, superpix_num0, superpix_allnum, conn_pattern,
        TrainSamLoc, TrainLabels, dist_mode2, center_mode2
    )
    # 假设superpix_img0, TrainSamLoc, TrainLabels和no_classes已经定义
    sp_num0 = np.max(superpix_img0)  # 计算superpix_img0的最大值
    initial_Labels = np.zeros((sp_num0, class_count))  # 创建一个全为零的矩阵

    for select_i in range(len(TrainSamLoc)):
        sp_idx = superpix_img0[TrainSamLoc[select_i]]  # 获取超像素索引
        initial_Labels[sp_idx - 1, TrainLabels[select_i] - 1] = 1  # 设置对应位置为1
    # 将初始标签矩阵转化为稀疏矩阵
    initial_Labels = csr_matrix(initial_Labels)

    N = WeightMat.shape[0]
    d = np.array(WeightMat.sum(axis=1)).flatten()  # 计算每行的和 (需要转换为数组)
    # 创建 Dinv 对角矩阵，使用 spdiags
    Dinv = spdiags(d, 0, N, N)
    # 计算Laplace矩阵
    Laplace_Mat_no_normalize2 = Dinv - WeightMat
    return Laplace_Mat_no_normalize2, initial_Labels, superpix_img0, sp_num0, Superpixel_mean_Features, WeightMat
