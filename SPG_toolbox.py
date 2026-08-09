import numpy as np
from scipy.sparse import triu, csr_matrix, spdiags, eye, diags
from scipy.linalg import inv
import scipy.sparse as sp

from sklearn.decomposition import PCA
from sklearn import preprocessing
from skimage.segmentation import slic,mark_boundaries,felzenszwalb,quickshift,random_walker

import matlab.engine
def f1DTo2DCoord(shape, idx):
    row, col = np.unravel_index(idx, shape)
    row, col = row + 1, col + 1
    return row, col


def SegmentsLabelProcess(labels):
    '''
    对labels做后处理，防止出现label不连续现象
    '''
    labels = np.array(labels, np.int64)
    H, W = labels.shape
    ls = list(set(np.reshape(labels, [-1]).tolist()))

    dic = {}
    for i in range(len(ls)):
        dic[ls[i]] = i

    new_labels = labels
    for i in range(H):
        for j in range(W):
            new_labels[i, j] = dic[new_labels[i, j]]
    return new_labels



def fPCA_FinalVersionV2(Data, Para, VarOrNum, Style):
    # 获取数据的维度
    DimNum = len(Data.shape)
    if DimNum == 3:
        Height, Width, Dim = Data.shape
        SamNum = Height * Width
        Data = Data.reshape(SamNum, Dim)
    else:
        SamNum, Dim = Data.shape

    # 计算协方差矩阵
    CoVar = np.cov(Data, rowvar=False)

    # 特征值分解
    eigvals, eigvecs = np.linalg.eigh(CoVar)

    # 特征值从大到小排序
    sorted_idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_idx]
    eigvecs = eigvecs[:, sorted_idx]

    if VarOrNum == 1:  # 如果根据方差能量比选择主成分
        EigenValueSum = np.sum(eigvals)
        TempEigenValueSum = 0
        EigenVectorNum = 0
        for i in range(Dim):
            TempEigenValueSum += eigvals[i]
            EigenValueWeight = TempEigenValueSum / EigenValueSum
            if EigenValueWeight > Para:
                break
            EigenVectorNum += 1
    else:  # 如果根据主成分个数选择
        EigenVectorNum = Para

    # 正则化处理
    if Style == 1:
        eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=0)

    # 提取主成分方向和方差
    PCDirection = eigvecs[:, :EigenVectorNum]
    PCVariance = eigvals[EigenVectorNum - 1]  # 第一个主成分的方差

    # 计算主成分
    PrimaryComponent = np.dot(Data, PCDirection)

    return PrimaryComponent, PCDirection, PCVariance


def superpixel_cutV2(HSI,n_segments = 1000,compactness = 20, max_iter = 20,
                             sigma = 0, min_size_factor = 0.3,max_size_factor = 2):
    # 数据standardization标准化,即提前全局BN
    height, width, bands = HSI.shape  # 原始高光谱数据的三个维度
    data = np.reshape(HSI, [height * width, bands])
    minMax = preprocessing.StandardScaler()
    data = minMax.fit_transform(data)
    img = np.reshape(data, [height, width, bands])

    # 执行 SLCI 并得到Q(nxm),S(m*b)
    (h, w, d) = img.shape
    # 计算超像素S以及相关系数矩阵Q
    # segments = slic(img, n_segments=self.n_segments, compactness=self.compactness, max_iter=self.max_iter,
    #                 convert2lab=False,sigma=self.sigma, enforce_connectivity=True,
    #                 min_size_factor=self.min_size_factor, max_size_factor=self.max_size_factor,slic_zero=False)

    segments = slic(img, n_segments, compactness, start_label=0)
    # segments = felzenszwalb(img, scale=1,sigma=0.5,min_size=25)
    # segments = quickshift(img,ratio=1,kernel_size=5,max_dist=4,sigma=0.8, convert2lab=False)
    # segments=LSC_superpixel(img,self.n_segments)
    # segments=SEEDS_superpixel(img,self.n_segments)

    # 判断超像素label是否连续,否则予以校正
    if segments.max() + 1 != len(list(set(np.reshape(segments, [-1]).tolist()))):
        segments = SegmentsLabelProcess(segments)
    superpixel_count = segments.max() + 1
    segments = np.reshape(segments+1, [-1])
    return segments

def superpixel_cut(HyperCube,n_segments = 1000,compactness = 0.1, max_iter = 20,
                             sigma = 0, min_size_factor = 0.3,max_size_factor = 2):

    # 假设 HyperCube 是一个 (nl, ns, nb) 形状的 3D 数组
    nl, ns, nb = HyperCube.shape
    # Step 1: 进行 PCA
    select_k = 20
    PC1,_,_ = fPCA_FinalVersionV2(HyperCube, select_k, 0, 0);
    # Step 2: 将 PCA 结果还原成图像格式
    HIM = PC1.reshape(nl, ns, select_k)
    # SLIC 超像素分割
    # 启动 MATLAB 引擎
    eng = matlab.engine.start_matlab()
    HyperCube = matlab.double(HIM.tolist())
    labels = eng.superpixel_cut(HyperCube,n_segments, nargout=1)
    segments = np.array(labels).flatten()
    eng.quit()
    # l = labels.reshape(ns, nl)
    # segments = np.reshape(l.T, (nl * ns, 1))

    # segments = np.reshape(segments+1, [-1])

    return segments

def WeightMatNormalization(WeightMat):
    N = WeightMat.shape[0]
    d = np.sum(WeightMat, axis=1)
    d = np.array(d).flatten()
    d[d != 0] = 1 / np.sqrt(d[d != 0])
    Dinv = diags(d)
    S_W = Dinv @ WeightMat @ Dinv
    S_W=csr_matrix(S_W);
    return S_W

def yl_label_reranging(TruthMap):
    TruthMap1D = TruthMap.flatten()
    UniqueLabel = np.unique(TruthMap1D)
    UniqueLabel = np.sort(UniqueLabel)

    for i in range(len(UniqueLabel)):
        TruthMap1D[TruthMap1D == UniqueLabel[i]] = i

    TruthMap = TruthMap1D.reshape(TruthMap.shape)
    return TruthMap

def SNG_sp_patch(ALL_data, spLocs, Height, Width):
    N, num_feature = ALL_data.shape  # N = 256, num_feature = 1100

    # Centering the data (subtract mean from each row)
    ALL_data_n = ALL_data - np.mean(ALL_data, axis=1, keepdims=True)

    # Calculate the L2 norm of each row (Euclidean distance)
    X1 = np.abs(np.sqrt(np.sum(ALL_data_n ** 2, axis=1)))

    # Compute pairwise distance using cosine similarity
    distance = np.abs(np.dot(ALL_data_n, ALL_data_n.T) / (np.outer(X1, X1)))

    # Initialize a boolean matrix for nodes to retain
    nodes_to_retain = np.zeros((N, N), dtype=bool)

    # Iterate over the locations of superpixels
    for i in range(len(spLocs)):
        spLoc_i = spLocs[i]

        # Convert 1D index to 2D coordinates
        TempYCoord, TempXCoord = f1DTo2DCoord((Height, Width), spLoc_i)

        # Define candidate coordinates around the current location
        candidateXCoord = [TempXCoord, TempXCoord, TempXCoord - 1, TempXCoord + 1]
        candidateYCoord = [TempYCoord - 1, TempYCoord + 1, TempYCoord, TempYCoord]

        # Compute candidate locations from coordinates
        candidateLocs = (np.array(candidateYCoord) - 1) * Width + np.array(candidateXCoord)-1

        # Find the indices of neighboring superpixels
        ######################################## method: for other data ########################################
        SNG_neighbors = np.isin(spLocs,candidateLocs)
        connectID=np.where(SNG_neighbors>0)[0]

        # Update nodes_to_retain for current location and its neighbors
        nodes_to_retain[i, connectID] = True

        ######################################## method: for WHU-Hi-HanChuan ########################################
        # SNG_neighbors = np.isin(candidateLocs, spLocs)
        # SNG_neighbors = candidateLocs[SNG_neighbors > 0]
        # # SNG_neighbors = SNG_neighbors[SNG_neighbors > 0]
        # nodes_to_retain[i, SNG_neighbors] = True

        nodes_to_retain[i, i] = False  # Diagonal should be zero

    # Make the matrix symmetric (i.e., undirected graph)
    nodes_to_retain = np.logical_or(nodes_to_retain, nodes_to_retain.T)

    # Initialize the adjacency matrix (Weight Matrix)
    WeightMat = np.zeros((N, N))

    # Assign distances to the corresponding positions in the weight matrix
    WeightMat[nodes_to_retain] = distance[nodes_to_retain]

    # Creating the adjacency matrix for training sample connections
    nodes_to_connect = np.zeros((N, N), dtype=bool)
    # Code for connecting nodes based on training samples (optional, commented out)
    # for train_i in range(len(TrainSamLoc)):
    #     itrainloc = TrainSamLoc[train_i]
    #     itrainlabel = TrainLabels[train_i]
    #     otherloc = TrainSamLoc[TrainLabels == itrainlabel]
    #     nodes_to_connect[itrainloc, otherloc] = True
    #     nodes_to_connect[otherloc, itrainloc] = True
    #     nodes_to_connect[itrainloc, itrainloc] = False

    # Set nodes to connect with weight 1
    WeightMat[nodes_to_connect] = 1

    # Convert to sparse matrix for efficiency
    WeightMat = csr_matrix(WeightMat)

    return WeightMat



def MultiGraph_Fusion_PCA_sub(S_W1, S_W2, flag_normalize=1):
    # Laplace_Mat = 0  # Not used in the function, so we can ignore it.

    SW_N = S_W1.shape[0]

    # Upper triangular parts of S_W1 and S_W2
    # S_W1 = np.triu(S_W1.toarray())
    # S_W2 = np.triu(S_W2.toarray())
    S_W1 = triu(S_W1)
    S_W2 = triu(S_W2)

    # 获取 S_W1 和 S_W2 中非零元素的二维坐标
    ind_SW1 = np.nonzero(S_W1.toarray())
    ind_SW2 = np.nonzero(S_W2.toarray())

    # 将二维坐标转换为一维索引
    ind_SW1_1d = np.ravel_multi_index(ind_SW1, S_W1.shape, order='F')
    ind_SW2_1d = np.ravel_multi_index(ind_SW2, S_W2.shape, order='F')

    # 获取两个集合的并集
    ind_SW = np.union1d(ind_SW1_1d, ind_SW2_1d)

    # 从 S_W1 和 S_W2 中提取对应位置的值
    vecSW1 = S_W1.toarray().flatten(order='F')[ind_SW]
    vecSW2 = S_W2.toarray().flatten(order='F')[ind_SW]

    # 合并两个列向量
    MultiLayerWeight = np.vstack([vecSW1, vecSW2]).T  # 两列合并成一个数组

    # 对合并的数组进行PCA降维
    # pca = PCA(n_components=1)
    # PCAWeightMat = pca.fit_transform(MultiLayerWeight)

    PCAWeightMat, _, _ = fPCA_FinalVersionV2(MultiLayerWeight, 1, 0, 0)

    # Initialize the sparse weight matrix S_W
    S_W = np.zeros((SW_N, SW_N))
    S_W = S_W.flatten(order='F')

    # Assign the PCA values to the sparse matrix (upper triangular part)
    S_W[ind_SW] = np.abs(PCAWeightMat.flatten(order='F'))

    S_W = S_W.reshape((SW_N, SW_N), order='F')

    # Make the matrix symmetric
    S_W = S_W + S_W.T

    # Convert to sparse matrix format
    S_W = csr_matrix(S_W)
    S_W_no_normalize = S_W.copy()

    # Normalize the weight matrix if needed
    if flag_normalize == 1:
        N = S_W.shape[0]
        d = np.array(S_W.sum(axis=1)).flatten()
        d[d != 0] = d[d != 0] ** (-0.5)

        Dinv = diags(d)
        S_W = Dinv.dot(S_W).dot(Dinv)
        print('execute normalization of SW mat')

    return S_W, S_W_no_normalize


def Connect_pixel_btwsp2(nodes_to_super, superpix_img, distance, param2, testaa, dist_mode1, center_mode1):
    N = distance.shape[0]
    nodes_to_add = np.zeros((N, N), dtype=bool)
    sp_num = nodes_to_super.shape[0]
    center_set = np.zeros(sp_num, dtype=int)

    # distance00 = distance[testaa][:, testaa]

    # Find center points for each superpixel
    for i in range(sp_num):
        pixel_set = np.where(superpix_img == i + 1)[0]
        tmp_distance = distance[pixel_set][:, pixel_set]

        # Number of nearest neighbors
        knn_param = 10

        # Calculating distances of k-nearest neighbors
        knn_distance = np.zeros(len(pixel_set))
        for kk in range(len(pixel_set)):
            # sort all possible neighbors according to distance
            if param2 in ['DIST', 'DIST2', 'DIST3']:
                temp = np.sort(distance[pixel_set[kk], :])
            else:
                if dist_mode1==1:
                    #higher acc
                    temp = np.sort(distance[kk, :])[::-1]
                else:
                    temp = np.sort(distance[pixel_set[kk], :])[::-1]

            # select k-th neighbor: knn_param+1, as the node itself is considered
            knn_distance[kk] = temp[knn_param]

        nodes_to_knn = np.ones((len(pixel_set), len(pixel_set)), dtype=bool)
        for kk in range(len(pixel_set)):
            if param2 in ['DIST', 'DIST2', 'DIST3']:
                nodes_to_knn[kk, tmp_distance[kk, :] > knn_distance[kk]] = False
            else:
                nodes_to_knn[kk, tmp_distance[kk, :] < knn_distance[kk]] = False

            nodes_to_knn[kk, kk] = False  # diagonal should be zero

        if center_mode1==1:
            # new version: higher acc
            center_point = np.argmax(np.sum(nodes_to_knn, axis=1))
        else:
            center_point = np.argmax(np.sum(nodes_to_knn, axis=0))
        center_set[i] = pixel_set[center_point]

    # Connect similar superpixels
    for i in range(sp_num):
        similar_set = np.where(nodes_to_super[i, :] > 0)[0]
        similar_set = np.append(similar_set, i)

        if len(similar_set) > 1:
            for i_set in range(1, len(similar_set)):
                for j_set in range(i_set):
                    point_idx1 = center_set[similar_set[i_set]]
                    point_idx2 = center_set[similar_set[j_set]]
                    nodes_to_add[point_idx1, point_idx2] = True
                    nodes_to_add[point_idx2, point_idx1] = True
        else:
            print(f'出现孤立超像素块,超像素块序号：{i + 1}')

    # Set diagonal to zero
    np.fill_diagonal(nodes_to_add, False)

    # Symmetrize the matrix
    nodes_to_add = nodes_to_add | nodes_to_add.T

    return nodes_to_add

def Connect_pixel_btwsp2_bigdata(nodes_to_super, superpix_img0, superpix_img, distance,
                                 param2, dist_mode2, center_mode2):
    N = distance.shape[0]
    nodes_to_add = np.zeros((N, N), dtype=bool)
    sp_num = nodes_to_super.shape[0]
    center_set = np.zeros(sp_num, dtype=int)

    # method1: max connectivity
    for i in range(sp_num):
        pixel_set = np.where(superpix_img == i+1)[0]
        sp0_idx = superpix_img0[pixel_set]-1
        sp0_idx = np.unique(sp0_idx)
        tmp_distance = distance[sp0_idx][:, sp0_idx]

        # Number of nearest neighbors
        knn_param = 10

        # Calculating distances of k-nearest neighbors
        knn_distance = np.zeros(len(sp0_idx))
        for kk in range(len(sp0_idx)):
            # sort all possible neighbors according to distance
            if param2 in ['DIST', 'DIST2', 'DIST3']:
                temp = np.sort(distance[kk, :])
            else:
                if dist_mode2==1:
                    # higher acc
                    temp = np.sort(distance[kk, :])[::-1]
                else:
                    temp = np.sort(distance[sp0_idx[kk], :])[::-1]

            # select k-th neighbor: knn_param+1, as the node itself is considered
            knn_distance[kk] = temp[knn_param].copy()

        nodes_to_knn = np.ones((len(sp0_idx), len(sp0_idx)), dtype=bool)
        for kk in range(len(sp0_idx)):
            if param2 in ['DIST', 'DIST2', 'DIST3']:
                nodes_to_knn[kk, tmp_distance[kk, :] > knn_distance[kk]] = False
            else:
                nodes_to_knn[kk, tmp_distance[kk, :] < knn_distance[kk]] = False

            nodes_to_knn[kk, kk] = False  # diagonal should be zero

        if center_mode2==0:
            # basic version:
            center_point = np.argmax(np.sum(nodes_to_knn, axis=0))
        elif center_mode2==1:
            # new version: higher acc
            center_point = np.argmax(np.sum(nodes_to_knn, axis=1))
        elif center_mode2 == 2:
            ## Indian Pines
            scores1 = np.sum(nodes_to_knn, axis=0)
            scores2 = np.sum(nodes_to_knn, axis=1)
            center_point = np.argmax(scores1.flatten()*5+scores2.flatten())
            # # center_point = np.argmax(scores1.flatten()+scores2.flatten())
        else:
            # basic version:
            center_point = np.argmax(np.sum(nodes_to_knn, axis=0))

        center_set[i] = sp0_idx[center_point].copy()

    # Connect similar superpixels
    for i in range(sp_num):
        similar_set = np.where(nodes_to_super[i, :] > 0)[0]
        similar_set = np.append(similar_set, i)

        if len(similar_set) > 1:
            for i_set in range(1, len(similar_set)):
                for j_set in range(i_set):
                    point_idx1 = center_set[similar_set[i_set]]
                    point_idx2 = center_set[similar_set[j_set]]
                    nodes_to_add[point_idx1, point_idx2] = True
                    nodes_to_add[point_idx2, point_idx1] = True
        else:
            print(f'出现孤立超像素块,超像素块序号：{i+1}')

    # Set diagonal to zero
    np.fill_diagonal(nodes_to_add, False)

    # Symmetrize the matrix
    nodes_to_add = nodes_to_add | nodes_to_add.T

    return nodes_to_add


def SuperPixel_Connect_Corr(superpix_img, Corr_Mat, flag2):
    sp_num = np.max(superpix_img)
    # Initialize distance matrix
    distance = np.zeros((sp_num, sp_num))

    # Compute pairwise distances (correlations between superpixels)
    for i in range(1, sp_num + 1):
        for j in range(1,i):
            sp_idx1 = np.where(superpix_img == i)[0]
            sp_idx2 = np.where(superpix_img == j)[0]
            distance_temp = Corr_Mat[sp_idx1,:][:,sp_idx2].copy()

            # You can use different methods to compute the distance between superpixels
            distance[i - 1, j-1] = np.max(distance_temp)

    # Complete the upper triangular part of the matrix
    distance = distance + distance.T
    N = sp_num

    # Number of nearest neighbors
    knn_param = 10

    # Calculating k-nearest neighbors' distances
    knn_distance = np.zeros(N)
    nn_distance = np.zeros(N)

    for i in range(N):
        temp = np.sort(distance[i])[::-1]
        knn_distance[i] = temp[knn_param].copy()  # k-th nearest neighbor
        nn_distance[i] = temp[1].copy()  # 1st nearest neighbor

    # Sparsification matrix based on k-nearest neighbors
    nodes_to_retain = np.ones((N, N), dtype=bool)

    for i in range(N):
        nodes_to_retain[i, distance[i, :] < knn_distance[i]] = False
        nodes_to_retain[i, i] = False  # Diagonal should be zero

    # Symmetric matrix: retain edges only if both directions are true
    # nodes_to_retain = np.logical_and(nodes_to_retain, nodes_to_retain.T)
    nodes_to_retain = nodes_to_retain & nodes_to_retain.T

    # Nearest neighbor matrix (nn_distance threshold)
    nodes_to_nn = np.ones((N, N), dtype=bool)

    for i in range(N):
        nodes_to_nn[i, distance[i, :] < nn_distance[i]] = False
        nodes_to_nn[i, i] = False  # Diagonal should be zero

    # Symmetric matrix: retain edges only if both directions are true
    # nodes_to_nn = np.logical_or(nodes_to_nn, nodes_to_nn.T)
    nodes_to_nn = nodes_to_nn | nodes_to_nn.T

    # Combine different types of connections based on the flag
    if flag2 == 'knn':
        nodes_to_retain = nodes_to_retain
    elif flag2 == 'nn':
        nodes_to_retain = nodes_to_nn
    elif flag2 == 'combine':
        nodes_to_retain = np.logical_or(nodes_to_retain, nodes_to_nn)
    return nodes_to_retain

    # if flag2 == 'knn':
    #     return nodes_to_retain
    # elif flag2 == 'nn':
    #     return nodes_to_nn
    # elif flag2 == 'combine':
    #     return nodes_to_retain | nodes_to_nn
    # else:
    #     raise ValueError(f"Invalid flag2 value: {flag2}")


def adjMat2ConnetMat(distance, flag2):
    N = distance.shape[0]
    # Number of nearest neighbors
    knn_param = 10

    # Calculating distances of k-nearest neighbors
    knn_distance = np.zeros(N)
    nn_distance = np.zeros(N)

    for i in range(N):
        # sort all possible neighbors according to distance
        temp = np.sort(distance[i, :])[::-1]
        # select k-th neighbor: knn_param+1, as the node itself is considered
        knn_distance[i] = temp[knn_param].copy()
        nn_distance[i] = temp[1].copy()

    # sparsification matrix
    nodes_to_retain = np.ones((N, N), dtype=bool)
    for i in range(N):
        nodes_to_retain[i, distance[i, :] < knn_distance[i]] = False
        nodes_to_retain[i, i] = False  # diagonal should be zero

    # Symmetrize the matrix
    nodes_to_retain = nodes_to_retain & nodes_to_retain.T

    # nearest neighbor matrix
    nodes_to_nn = np.ones((N, N), dtype=bool)
    for i in range(N):
        nodes_to_nn[i, distance[i, :] < nn_distance[i]] = False
        nodes_to_nn[i, i] = False  # diagonal should be zero

    # Symmetrize the matrix
    nodes_to_nn = nodes_to_nn | nodes_to_nn.T

    # Select matrix based on flag2
    if flag2 == 'knn':
        return nodes_to_retain
    elif flag2 == 'nn':
        return nodes_to_nn
    elif flag2 == 'combine':
        return nodes_to_retain | nodes_to_nn
    else:
        raise ValueError(f"Invalid flag2 value: {flag2}")

def SuperPixWeightMat_Experiment_with_Block2Block(HyperCube, superpix_img, param2, flag2, dist_mode1, center_mode1):
    sp_num = np.max(superpix_img)
    Height, Width, Bands = HyperCube.shape
    ALL_data = HyperCube.reshape(-1, Bands)  # Flattening the 3D array into 2D
    N, num_feature = ALL_data.shape

    # Normalize the data by subtracting the mean
    ALL_data_n = ALL_data - np.mean(ALL_data, axis=1, keepdims=True)

    # Compute distances
    X1 = np.sqrt(np.sum(ALL_data_n ** 2, axis=1))
    distance = np.abs(np.dot(ALL_data_n, ALL_data_n.T) / np.outer(X1, X1))

    # Number of nearest neighbors
    knn_param = 10

    # Calculate knn distances
    knn_distance = np.zeros(N)
    nn_distance = np.zeros(N)

    for i in range(N):
        temp = np.sort(distance[i])[::-1]
        knn_distance[i] = temp[knn_param]
        nn_distance[i] = temp[1]

    # Calculate sigma
    sigma = (1 / 3) * np.mean(knn_distance)

    nodes_to_retain = np.zeros((N, N), dtype=bool)

    for i in range(1, sp_num + 1):
        sp_idx1 = np.where(superpix_img == i)[0]

        for pixel_idx in sp_idx1:
            TempYCoord, TempXCoord = f1DTo2DCoord((Height, Width), pixel_idx)

            # Determine boundaries
            if (pixel_idx+1) % Width == 0:
                lower_x = -1
                upper_x = 0
            elif (pixel_idx) % Width == 0:
                lower_x = 0
                upper_x = 1
            else:
                lower_x = -1
                upper_x = 1

            PointLocSet = []

            for y_j in range(-1, 2):
                PointLocTemp = (TempYCoord - 1 + y_j) * Width + TempXCoord - 1
                if PointLocTemp in sp_idx1:
                    PointLocSet.append(PointLocTemp)

            for x_i in range(lower_x, upper_x + 1):
                PointLocTemp = (TempYCoord - 1) * Width + TempXCoord + x_i - 1
                if PointLocTemp in sp_idx1:
                    PointLocSet.append(PointLocTemp)

            if PointLocSet:
                # Adjust for 0-based indexing
                nodes_to_retain[pixel_idx, np.array(PointLocSet)] = True
            # nodes_to_retain[pixel_idx, PointLocSet] = True
            nodes_to_retain[pixel_idx, pixel_idx] = False  # Diagonal should be zero

    nodes_to_retain |= nodes_to_retain.T

    # Different superpixel block connection strategies
    if param2 in ['DIST', 'DIST2', 'DIST3']:
        nodes_to_super = SuperPixel_Connect_Distance(HyperCube, superpix_img, flag2)
    else:
        nodes_to_super = SuperPixel_Connect_Corr(superpix_img, distance, flag2)

    testaa = np.arange(0, Height * Width)
    testaa = testaa.reshape(Height, Width)
    testaa = testaa.T
    testaa = testaa.ravel()
    superpix_img00 = superpix_img[testaa]
    distance00 = distance[testaa][:, testaa]
    nodes_to_add2 = Connect_pixel_btwsp2(nodes_to_super, superpix_img00, distance00,
                                         param2, testaa, dist_mode1, center_mode1)
    nodes_to_add = np.zeros_like(nodes_to_add2)
    nodes_to_add[testaa][:, testaa] = nodes_to_add2
    nodes_to_add = nodes_to_add2[testaa][:, testaa]
    nodes_to_retain |= nodes_to_add
    nodes_to_retain |= nodes_to_retain.T

    nodes_to_add_sparse = csr_matrix(nodes_to_add)

    # Create adjacency matrix
    WeightMat = np.zeros((N, N))
    WeightMat[nodes_to_retain] = distance[nodes_to_retain]
    WeightMat_sparse = csr_matrix(WeightMat)

    return WeightMat_sparse, nodes_to_add_sparse


def SuperPixWeightMat_Experiment_with_Block2Block_bigdata(Superpixel_mean_Features, HyperCube, superpix_img0,
                                                          superpix_img, param2, flag2, dist_mode2, center_mode2):
    sp_num = np.max(superpix_img)
    Height, Width, Bands = HyperCube.shape
    ALL_data = HyperCube.reshape(-1, Bands)  # Flattening the 3D array into 2D

    Superpixel_mean_Features1 = []
    for i in range(1, sp_num + 1):
        sp_idx1 = np.where(superpix_img == i)[0]
        X0 = np.mean(ALL_data[sp_idx1, :], axis=0)
        Superpixel_mean_Features1.append(X0)

    Superpixel_mean_Features1 = np.array(Superpixel_mean_Features1)

    ALL_data = Superpixel_mean_Features.copy()
    N, num_feature = ALL_data.shape

    # Normalize the data by subtracting the mean
    ALL_data_n = ALL_data - np.mean(ALL_data, axis=1, keepdims=True)

    # Compute distances
    X1 = np.sqrt(np.sum(ALL_data_n ** 2, axis=1))
    distance = np.abs(np.dot(ALL_data_n, ALL_data_n.T) / np.outer(X1, X1))


    ALL_data_n2=Superpixel_mean_Features1-np.mean(Superpixel_mean_Features1, axis=1, keepdims=True)
    X2=np.sqrt(np.sum(ALL_data_n2 ** 2, axis=1))
    distance1 = np.abs(np.dot(ALL_data_n2, ALL_data_n2.T) / np.outer(X2, X2))

    # Number of nearest neighbors
    knn_param = 10

    # Calculate knn distances
    knn_distance = np.zeros(N)
    nn_distance = np.zeros(N)

    for i in range(N):
        temp = np.sort(distance[i])[::-1]
        knn_distance[i] = temp[knn_param].copy()
        nn_distance[i] = temp[1].copy()

    # Calculate sigma
    sigma = (1 / 3) * np.mean(knn_distance)

    nodes_to_retain = np.zeros((N, N), dtype=bool)

    for i in range(1, sp_num + 1):
        sp_idx1 = np.where(superpix_img == i)[0]

        for idxii in range(len(sp_idx1)):
            pixel_idx = sp_idx1[idxii]
            sp0_idx = superpix_img0[pixel_idx]
            PointLocSet = []

            # Calculate coordinate using unravel_index
            TempYCoord, TempXCoord = f1DTo2DCoord((Height, Width), pixel_idx)

            # Determine boundaries
            if (pixel_idx+1) % Width == 0:
                lower_x = -1
                upper_x = 0
            elif (pixel_idx) % Width == 0:
                lower_x = 0
                upper_x = 1
            else:
                lower_x = -1
                upper_x = 1

            # Check y-axis neighboring pixels
            for y_j in range(-1, 2):
                # PointLocTemp = (TempXCoord - 1) * Height + (TempYCoord + y_j) - 1
                PointLocTemp = (TempYCoord - 1 + y_j) * Width + TempXCoord - 1
                if PointLocTemp in sp_idx1:
                    SP0LocTemp = superpix_img0[PointLocTemp]
                    SP0LocTemp = np.unique(SP0LocTemp)
                    SP0LocTemp = SP0LocTemp[SP0LocTemp != sp0_idx]
                    PointLocSet.extend(SP0LocTemp)

            # Check x-axis neighboring pixels
            for x_i in range(lower_x, upper_x + 1):
                # PointLocTemp = (TempXCoord - 1) * Height + TempYCoord - 1
                PointLocTemp = (TempYCoord - 1) * Width + TempXCoord + x_i - 1
                if PointLocTemp in sp_idx1:
                    SP0LocTemp = superpix_img0[PointLocTemp]
                    SP0LocTemp = np.unique(SP0LocTemp)
                    SP0LocTemp = SP0LocTemp[SP0LocTemp != sp0_idx]
                    PointLocSet.extend(SP0LocTemp)

            # Remove duplicates
            PointLocSet = list(set(PointLocSet))

            if PointLocSet:
                # Adjust for 0-based indexing
                nodes_to_retain[sp0_idx - 1, np.array(PointLocSet) - 1] = True

            # Set nodes to retain
            # nodes_to_retain[sp0_idx-1, PointLocSet-1] = True
            nodes_to_retain[sp0_idx-1, sp0_idx-1] = False  # diagonal should be zero

    nodes_to_retain |= nodes_to_retain.T

    # Different superpixel block connection strategies
    nodes_to_super = adjMat2ConnetMat(distance1, flag2);
    nodes_to_add = Connect_pixel_btwsp2_bigdata(nodes_to_super, superpix_img0, superpix_img,
                                                distance, param2, dist_mode2, center_mode2)
    nodes_to_retain |= nodes_to_add
    nodes_to_retain |= nodes_to_retain.T

    nodes_to_add_sparse = csr_matrix(nodes_to_add)

    # Create adjacency matrix
    WeightMat = np.zeros((N, N))
    WeightMat[nodes_to_retain] = distance[nodes_to_retain]
    WeightMat_sparse = csr_matrix(WeightMat)

    return WeightMat_sparse, nodes_to_add_sparse



def FSPG_graph_construction(HyperCube, superpix_allnum, conn_pattern, TrainSamLoc, TrainLabels, dist_mode1, center_mode1):
    Height, Width, Bands = HyperCube.shape
    HyperCube_vec = HyperCube.reshape(Width * Height, Bands)  # 3D to 2D
    S_W = csr_matrix([])
    fuse_idx = 0

    for superpix_num in superpix_allnum:
        # Superpixel segmentation
        superpix_img = superpixel_cut(HyperCube, superpix_num)
        sp_num = np.max(superpix_img)

        for select_i in range(1, sp_num + 1):
            pos_idx = np.where(superpix_img == select_i)[0]
            spTrainflag = np.isin(TrainSamLoc, pos_idx)
            spTrainSamLoc = TrainSamLoc[spTrainflag]
            spTrainLabels = TrainLabels[spTrainflag]

            if len(spTrainSamLoc) > 0:
                spTrainLabels = yl_label_reranging(spTrainLabels) + 1
                if np.max(spTrainLabels) > 1:
                    # Compute weight matrix for superpixels
                    WeightMat_sp = SNG_sp_patch(HyperCube_vec[pos_idx, :], pos_idx, Height, Width)
                    N_sp = WeightMat_sp.shape[0]
                    d = np.array(WeightMat_sp.sum(axis=1)).flatten()
                    d[d != 0] = 1 / np.sqrt(d[d != 0])
                    Dinv_sp = spdiags(d, 0, N_sp, N_sp)
                    SW_sp = Dinv_sp @ WeightMat_sp @ Dinv_sp
                    initial_Labels_sp = np.zeros((N_sp, np.max(spTrainLabels)))

                    for spTr_i in range(len(spTrainSamLoc)):
                        tmp_spTrainSamLoc = spTrainSamLoc[spTr_i]
                        pos_idx_fortmpsp = np.where(pos_idx == tmp_spTrainSamLoc)[0]
                        initial_Labels_sp[pos_idx_fortmpsp, spTrainLabels[spTr_i] - 1] = 1

                    initial_Labels_sp = csr_matrix(initial_Labels_sp)
                    para_alpha = 0.1
                    UU = eye(N_sp) - para_alpha * SW_sp
                    # F = inv(UU) @ initial_Labels_sp  # F = (I - alpha * S)^(-1) * Y
                    F = sp.linalg.spsolve(UU, initial_Labels_sp)
                    PredLabels_sp = np.argmax(F.toarray(), axis=1)

                    # Update superpixel labels based on predictions
                    for splabel_i in range(1, np.max(PredLabels_sp) + 1):
                        sploc_i = np.where(PredLabels_sp == splabel_i)[0]
                        max_sp_ind = np.max(superpix_img)
                        superpix_img[pos_idx[sploc_i]] = max_sp_ind + 1

        superpix_img = yl_label_reranging(superpix_img) + 1

        # Compute weight matrix for superpixels
        if superpix_num > 11:
            WeightMat, nodes_to_add_tmp = SuperPixWeightMat_Experiment_with_Block2Block(
                HyperCube, superpix_img, 'CORR', conn_pattern, dist_mode1, center_mode1
            )
            S_W_tmp = WeightMatNormalization(WeightMat)

        if S_W.nnz==0:
            S_W = csr_matrix(S_W_tmp)
            nodes_to_add = nodes_to_add_tmp
            S_W_no_normalize = WeightMat
        else:
            if superpix_num == superpix_allnum[-1]:

                S_W, S_W_no_normalize = MultiGraph_Fusion_PCA_sub(S_W_tmp, S_W)
            else:
                S_W, S_W_no_normalize = MultiGraph_Fusion_PCA_sub(S_W_tmp, S_W)
            # nodes_to_add = nodes_to_add | nodes_to_add_tmp
            nodes_to_add = nodes_to_add.maximum(nodes_to_add_tmp)

    # Incorporate training sample locations and labels
    for train_i in range(len(TrainSamLoc)):
        itrainloc = TrainSamLoc[train_i]
        itrainlabel = TrainLabels[train_i]
        otherloc = TrainSamLoc[TrainLabels == itrainlabel]

        S_W_no_normalize[itrainloc, otherloc] = 1
        S_W_no_normalize[otherloc, itrainloc] = 1
        S_W_no_normalize[itrainloc, itrainloc] = 0

    return S_W_no_normalize


def FSPG_graph_construction251103(HyperCube, superpix_allnum, conn_pattern, TrainSamLoc, TrainLabels, dist_mode1, center_mode1):
    Height, Width, Bands = HyperCube.shape
    HyperCube_vec = HyperCube.reshape(Width * Height, Bands)  # 3D to 2D
    S_W = csr_matrix([])
    fuse_idx = 0

    for superpix_num in superpix_allnum:
        # Superpixel segmentation
        superpix_img = superpixel_cut(HyperCube, superpix_num)
        sp_num = np.max(superpix_img)

        for select_i in range(1, sp_num + 1):
            pos_idx = np.where(superpix_img == select_i)[0]
            spTrainflag = np.isin(TrainSamLoc, pos_idx)
            spTrainSamLoc = TrainSamLoc[spTrainflag]
            spTrainLabels = TrainLabels[spTrainflag]

            if len(spTrainSamLoc) > 0:
                spTrainLabels = yl_label_reranging(spTrainLabels) + 1
                if np.max(spTrainLabels) > 1:
                    # Compute weight matrix for superpixels
                    WeightMat_sp = SNG_sp_patch(HyperCube_vec[pos_idx, :], pos_idx, Height, Width)
                    N_sp = WeightMat_sp.shape[0]
                    d = np.array(WeightMat_sp.sum(axis=1)).flatten()
                    d[d != 0] = 1 / np.sqrt(d[d != 0])
                    # 使用逐元素乘法，避免矩阵乘法
                    SW_sp = WeightMat_sp.multiply(d.reshape(-1, 1)).multiply(d)
                    initial_Labels_sp = np.zeros((N_sp, np.max(spTrainLabels)))

                    for spTr_i in range(len(spTrainSamLoc)):
                        tmp_spTrainSamLoc = spTrainSamLoc[spTr_i]
                        pos_idx_fortmpsp = np.where(pos_idx == tmp_spTrainSamLoc)[0]
                        initial_Labels_sp[pos_idx_fortmpsp, spTrainLabels[spTr_i] - 1] = 1

                    initial_Labels_sp = csr_matrix(initial_Labels_sp)
                    para_alpha = 0.5
                    UU = eye(N_sp) - para_alpha * SW_sp
                    # F = inv(UU) @ initial_Labels_sp
                    # from scipy.sparse.linalg import spsolve
                    # # 求解 F = inv(UU) @ initial_Labels_sp 等价于解 UU @ F = initial_Labels_sp
                    # F = spsolve(UU, initial_Labels_sp)
                    UU_dense = UU.toarray()
                    initial_Labels_dense = initial_Labels_sp.toarray()
                    # 使用numpy的线性代数求解
                    F = inv(UU_dense) @ initial_Labels_dense
                    # F = np.linalg.solve(UU_dense, initial_Labels_dense)
                    F = csr_matrix(F)
                    PredLabels_sp = np.argmax(F.toarray(), axis=1)

                    # Update superpixel labels based on predictions
                    for splabel_i in range(1, np.max(PredLabels_sp) + 1):
                        sploc_i = np.where(PredLabels_sp == splabel_i)[0]
                        max_sp_ind = np.max(superpix_img)
                        superpix_img[pos_idx[sploc_i]] = max_sp_ind + 1

        superpix_img = yl_label_reranging(superpix_img) + 1

        # Compute weight matrix for superpixels
        if superpix_num > 11:
            WeightMat, nodes_to_add_tmp = SuperPixWeightMat_Experiment_with_Block2Block(
                HyperCube, superpix_img, 'CORR', conn_pattern, dist_mode1, center_mode1
            )
            S_W_tmp = WeightMatNormalization(WeightMat)

        if S_W.nnz==0:
            S_W = csr_matrix(S_W_tmp)
            nodes_to_add = nodes_to_add_tmp
            S_W_no_normalize = WeightMat
        else:
            if superpix_num == superpix_allnum[-1]:

                S_W, S_W_no_normalize = MultiGraph_Fusion_PCA_sub(S_W_tmp, S_W)
            else:
                S_W, S_W_no_normalize = MultiGraph_Fusion_PCA_sub(S_W_tmp, S_W)
            # nodes_to_add = nodes_to_add | nodes_to_add_tmp
            nodes_to_add = nodes_to_add.maximum(nodes_to_add_tmp)

    # Incorporate training sample locations and labels
    for train_i in range(len(TrainSamLoc)):
        itrainloc = TrainSamLoc[train_i]
        itrainlabel = TrainLabels[train_i]
        otherloc = TrainSamLoc[TrainLabels == itrainlabel]

        S_W_no_normalize[itrainloc, otherloc] = 1
        S_W_no_normalize[otherloc, itrainloc] = 1
        S_W_no_normalize[itrainloc, itrainloc] = 0

    return S_W_no_normalize

def SiSGs_bigdata_ablation251128(HyperCube, superpix_num0, superpix_allnum, conn_pattern, TrainSamLoc, TrainLabels):
    Height, Width, Bands = HyperCube.shape
    HyperCube_vec = HyperCube.reshape(Width * Height, Bands)  # 3D to 2D
    S_W = csr_matrix([])

    # Step 1: Superpixel segmentation for superpix_num0
    superpix_img0 = superpixel_cut(HyperCube, superpix_num0)
    sp_num = np.max(superpix_img0)

    # 同一超像素内不同标签分裂
    for select_i in range(1, sp_num + 1):
        pos_idx = np.where(superpix_img0 == select_i)[0]
        spTrainflag = np.isin(TrainSamLoc, pos_idx)
        spTrainSamLoc = TrainSamLoc[spTrainflag]
        spTrainLabels = TrainLabels[spTrainflag]

        if len(spTrainSamLoc) > 0:
            spTrainLabels = yl_label_reranging(spTrainLabels) + 1
            if np.max(spTrainLabels) > 1:
                # Compute weight matrix for superpixels
                WeightMat_sp = SNG_sp_patch(HyperCube_vec[pos_idx, :], pos_idx, Height, Width)
                N_sp = WeightMat_sp.shape[0]
                # d = np.sum(WeightMat_sp, axis=1)
                # d = np.array(d)
                # d[d != 0] = 1 / np.sqrt(d[d != 0])
                # d = csr_matrix(d)
                # Dinv_sp = spdiags(d, 0, N_sp, N_sp)
                # SW_sp = Dinv_sp @ WeightMat_sp @ Dinv_sp
                d = np.array(WeightMat_sp.sum(axis=1)).flatten()
                d[d != 0] = 1 / np.sqrt(d[d != 0])

                # 使用逐元素乘法，避免矩阵乘法
                SW_sp = WeightMat_sp.multiply(d.reshape(-1, 1)).multiply(d)
                initial_Labels_sp = np.zeros((N_sp, np.max(spTrainLabels)))

                for spTr_i in range(len(spTrainSamLoc)):
                    tmp_spTrainSamLoc = spTrainSamLoc[spTr_i]
                    pos_idx_fortmpsp = np.where(pos_idx == tmp_spTrainSamLoc)[0]
                    initial_Labels_sp[pos_idx_fortmpsp, spTrainLabels[spTr_i] - 1] = 1

                initial_Labels_sp = csr_matrix(initial_Labels_sp)
                para_alpha = 0.5
                UU = eye(N_sp) - para_alpha * SW_sp
                # F = inv(UU) @ initial_Labels_sp
                # from scipy.sparse.linalg import spsolve
                # # 求解 F = inv(UU) @ initial_Labels_sp 等价于解 UU @ F = initial_Labels_sp
                # F = spsolve(UU, initial_Labels_sp)
                UU_dense = UU.toarray()
                initial_Labels_dense = initial_Labels_sp.toarray()
                # 使用numpy的线性代数求解
                F = inv(UU_dense) @ initial_Labels_dense
                # F = np.linalg.solve(UU_dense, initial_Labels_dense)
                F = csr_matrix(F)
                # F = (I - alpha * S)^(-1) * Y
                PredLabels_sp = np.argmax(F.toarray(), axis=1)

                # Update superpixel labels based on predictions
                for splabel_i in range(1, np.max(PredLabels_sp) + 1):
                    sploc_i = np.where(PredLabels_sp == splabel_i)[0]
                    max_sp_ind = np.max(superpix_img0)
                    superpix_img0[pos_idx[sploc_i]] = max_sp_ind + 1

    superpix_img0 = yl_label_reranging(superpix_img0) + 1

    # Compute Superpixel Mean Features
    sp_num0 = np.max(superpix_img0)
    ALL_data = HyperCube.reshape(Width * Height, Bands)  # 3D to 2D
    Superpixel_mean_Features = []

    for i in range(0, sp_num0):
        sp_idx1 = np.where(superpix_img0 == i+1)[0]
        X1 = np.mean(ALL_data[sp_idx1, :], axis=0)
        Superpixel_mean_Features.append(X1)

    Superpixel_mean_Features = np.array(Superpixel_mean_Features)

    # Step 2: Iterate through different superpixel numbers
    for superpix_num in superpix_allnum:
        # if superpix_num==900:
        #     superpix_img = superpix_img01
        # if superpix_num==1800:
        #     superpix_img = superpix_img02

        # Superpixel segmentation
        superpix_img = superpixel_cut(HyperCube, superpix_num)
        sp_num = np.max(superpix_img)

        for select_i in range(1, sp_num + 1):
            pos_idx = np.where(superpix_img == select_i)[0]
            spTrainflag = np.isin(TrainSamLoc, pos_idx)
            spTrainSamLoc = TrainSamLoc[spTrainflag]
            spTrainLabels = TrainLabels[spTrainflag]

            if len(spTrainSamLoc) > 0:
                spTrainLabels = yl_label_reranging(spTrainLabels) + 1
                if np.max(spTrainLabels) > 1:
                    # Compute weight matrix for superpixels
                    WeightMat_sp = SNG_sp_patch(HyperCube_vec[pos_idx, :], pos_idx, Height, Width)
                    N_sp = WeightMat_sp.shape[0]
                    d = np.array(WeightMat_sp.sum(axis=1)).flatten()
                    d[d != 0] = 1 / np.sqrt(d[d != 0])

                    # 使用逐元素乘法，避免矩阵乘法
                    SW_sp = WeightMat_sp.multiply(d.reshape(-1, 1)).multiply(d)
                    initial_Labels_sp = np.zeros((N_sp, np.max(spTrainLabels)))

                    for spTr_i in range(len(spTrainSamLoc)):
                        tmp_spTrainSamLoc = spTrainSamLoc[spTr_i]
                        pos_idx_fortmpsp = np.where(pos_idx == tmp_spTrainSamLoc)[0]
                        initial_Labels_sp[pos_idx_fortmpsp, spTrainLabels[spTr_i] - 1] = 1

                    initial_Labels_sp = csr_matrix(initial_Labels_sp)
                    para_alpha = 0.5
                    UU = eye(N_sp) - para_alpha * SW_sp
                    # other data
                    UU_dense = UU.toarray()
                    initial_Labels_dense = initial_Labels_sp.toarray()
                    # 使用numpy的线性代数求解
                    F = inv(UU_dense) @ initial_Labels_dense
                    # F = np.linalg.solve(UU_dense, initial_Labels_dense)
                    F = csr_matrix(F)
                    # F = sp.linalg.spsolve(UU, initial_Labels_sp)
                    PredLabels_sp = np.argmax(F.toarray(), axis=1)

                    # Update superpixel labels based on predictions
                    for splabel_i in range(1, np.max(PredLabels_sp) + 1):
                        sploc_i = np.where(PredLabels_sp == splabel_i)[0]
                        max_sp_ind = np.max(superpix_img)
                        superpix_img[pos_idx[sploc_i]] = max_sp_ind + 1

        superpix_img = yl_label_reranging(superpix_img) + 1

        # Compute weight matrix based on superpixel number
        if superpix_num > 11:
            WeightMat, nodes_to_add_tmp = SuperPixWeightMat_Experiment_with_Block2Block_bigdata(
                Superpixel_mean_Features, HyperCube, superpix_img0, superpix_img,
                'CORR', conn_pattern, dist_mode2, center_mode2
            )
            S_W_tmp = WeightMatNormalization(WeightMat)
        #
        if S_W.nnz==0:
            S_W = csr_matrix(S_W_tmp)

    # Step 3: Incorporate training sample locations and labels
    for train_i in range(len(TrainSamLoc)):
        itrainloc = TrainSamLoc[train_i]
        sp0_idx1 = superpix_img0[itrainloc]-1
        itrainlabel = TrainLabels[train_i]
        otherloc = TrainSamLoc[TrainLabels == itrainlabel]
        sp0_idx2 = superpix_img0[otherloc]-1
        sp0_idx2 = np.unique(sp0_idx2)

        S_W[sp0_idx1, sp0_idx2] = 1
        S_W[sp0_idx2, sp0_idx1] = 1
        S_W[sp0_idx1, sp0_idx1] = 0

        S_W_tmp[sp0_idx1, sp0_idx2] = 1
        S_W_tmp[sp0_idx2, sp0_idx1] = 1
        S_W_tmp[sp0_idx1, sp0_idx1] = 0

    return S_W, S_W_tmp, superpix_img0, sp_num0

def FSPG_graph_construction_bigdata251103(HyperCube, superpix_num0, superpix_allnum, conn_pattern,
                                          TrainSamLoc, TrainLabels, dist_mode2, center_mode2, knn_param=10
                                    ):
    Height, Width, Bands = HyperCube.shape
    HyperCube_vec = HyperCube.reshape(Width * Height, Bands)  # 3D to 2D
    S_W = csr_matrix([])

    # Step 1: Superpixel segmentation for superpix_num0
    superpix_img0 = superpixel_cut(HyperCube, superpix_num0)
    sp_num = np.max(superpix_img0)

    # 同一超像素内不同标签分裂
    for select_i in range(1, sp_num + 1):
        pos_idx = np.where(superpix_img0 == select_i)[0]
        spTrainflag = np.isin(TrainSamLoc, pos_idx)
        spTrainSamLoc = TrainSamLoc[spTrainflag]
        spTrainLabels = TrainLabels[spTrainflag]

        if len(spTrainSamLoc) > 0:
            spTrainLabels = yl_label_reranging(spTrainLabels) + 1
            if np.max(spTrainLabels) > 1:
                # Compute weight matrix for superpixels
                WeightMat_sp = SNG_sp_patch(HyperCube_vec[pos_idx, :], pos_idx, Height, Width)
                N_sp = WeightMat_sp.shape[0]
                # d = np.sum(WeightMat_sp, axis=1)
                # d = np.array(d)
                # d[d != 0] = 1 / np.sqrt(d[d != 0])
                # d = csr_matrix(d)
                # Dinv_sp = spdiags(d, 0, N_sp, N_sp)
                # SW_sp = Dinv_sp @ WeightMat_sp @ Dinv_sp
                d = np.array(WeightMat_sp.sum(axis=1)).flatten()
                d[d != 0] = 1 / np.sqrt(d[d != 0])

                # 使用逐元素乘法，避免矩阵乘法
                SW_sp = WeightMat_sp.multiply(d.reshape(-1, 1)).multiply(d)
                initial_Labels_sp = np.zeros((N_sp, np.max(spTrainLabels)))

                for spTr_i in range(len(spTrainSamLoc)):
                    tmp_spTrainSamLoc = spTrainSamLoc[spTr_i]
                    pos_idx_fortmpsp = np.where(pos_idx == tmp_spTrainSamLoc)[0]
                    initial_Labels_sp[pos_idx_fortmpsp, spTrainLabels[spTr_i] - 1] = 1

                initial_Labels_sp = csr_matrix(initial_Labels_sp)
                para_alpha = 0.5
                UU = eye(N_sp) - para_alpha * SW_sp
                # F = inv(UU) @ initial_Labels_sp
                # from scipy.sparse.linalg import spsolve
                # # 求解 F = inv(UU) @ initial_Labels_sp 等价于解 UU @ F = initial_Labels_sp
                # F = spsolve(UU, initial_Labels_sp)
                UU_dense = UU.toarray()
                initial_Labels_dense = initial_Labels_sp.toarray()
                # 使用numpy的线性代数求解
                F = inv(UU_dense) @ initial_Labels_dense
                # F = np.linalg.solve(UU_dense, initial_Labels_dense)
                F = csr_matrix(F)
                # F = (I - alpha * S)^(-1) * Y
                PredLabels_sp = np.argmax(F.toarray(), axis=1)

                # Update superpixel labels based on predictions
                for splabel_i in range(1, np.max(PredLabels_sp) + 1):
                    sploc_i = np.where(PredLabels_sp == splabel_i)[0]
                    max_sp_ind = np.max(superpix_img0)
                    superpix_img0[pos_idx[sploc_i]] = max_sp_ind + 1

    superpix_img0 = yl_label_reranging(superpix_img0) + 1

    # Compute Superpixel Mean Features
    sp_num0 = np.max(superpix_img0)
    ALL_data = HyperCube.reshape(Width * Height, Bands)  # 3D to 2D
    Superpixel_mean_Features = []

    for i in range(0, sp_num0):
        sp_idx1 = np.where(superpix_img0 == i+1)[0]
        X1 = np.mean(ALL_data[sp_idx1, :], axis=0)
        Superpixel_mean_Features.append(X1)

    Superpixel_mean_Features = np.array(Superpixel_mean_Features)

    # Step 2: Iterate through different superpixel numbers
    for superpix_num in superpix_allnum:
        # if superpix_num==900:
        #     superpix_img = superpix_img01
        # if superpix_num==1800:
        #     superpix_img = superpix_img02

        # Superpixel segmentation
        superpix_img = superpixel_cut(HyperCube, superpix_num)
        sp_num = np.max(superpix_img)

        for select_i in range(1, sp_num + 1):
            pos_idx = np.where(superpix_img == select_i)[0]
            spTrainflag = np.isin(TrainSamLoc, pos_idx)
            spTrainSamLoc = TrainSamLoc[spTrainflag]
            spTrainLabels = TrainLabels[spTrainflag]

            if len(spTrainSamLoc) > 0:
                spTrainLabels = yl_label_reranging(spTrainLabels) + 1
                if np.max(spTrainLabels) > 1:
                    # Compute weight matrix for superpixels
                    WeightMat_sp = SNG_sp_patch(HyperCube_vec[pos_idx, :], pos_idx, Height, Width)
                    N_sp = WeightMat_sp.shape[0]
                    d = np.array(WeightMat_sp.sum(axis=1)).flatten()
                    d[d != 0] = 1 / np.sqrt(d[d != 0])

                    # 使用逐元素乘法，避免矩阵乘法
                    SW_sp = WeightMat_sp.multiply(d.reshape(-1, 1)).multiply(d)
                    initial_Labels_sp = np.zeros((N_sp, np.max(spTrainLabels)))

                    for spTr_i in range(len(spTrainSamLoc)):
                        tmp_spTrainSamLoc = spTrainSamLoc[spTr_i]
                        pos_idx_fortmpsp = np.where(pos_idx == tmp_spTrainSamLoc)[0]
                        initial_Labels_sp[pos_idx_fortmpsp, spTrainLabels[spTr_i] - 1] = 1

                    initial_Labels_sp = csr_matrix(initial_Labels_sp)
                    para_alpha = 0.5
                    UU = eye(N_sp) - para_alpha * SW_sp
                    # other data
                    UU_dense = UU.toarray()
                    initial_Labels_dense = initial_Labels_sp.toarray()
                    # 使用numpy的线性代数求解
                    F = inv(UU_dense) @ initial_Labels_dense
                    # F = np.linalg.solve(UU_dense, initial_Labels_dense)
                    F = csr_matrix(F)
                    # F = sp.linalg.spsolve(UU, initial_Labels_sp)
                    PredLabels_sp = np.argmax(F.toarray(), axis=1)

                    # Update superpixel labels based on predictions
                    for splabel_i in range(1, np.max(PredLabels_sp) + 1):
                        sploc_i = np.where(PredLabels_sp == splabel_i)[0]
                        max_sp_ind = np.max(superpix_img)
                        superpix_img[pos_idx[sploc_i]] = max_sp_ind + 1

        superpix_img = yl_label_reranging(superpix_img) + 1

        # Compute weight matrix based on superpixel number
        if superpix_num > 11:
            WeightMat, nodes_to_add_tmp = SuperPixWeightMat_Experiment_with_Block2Block_bigdata(
                Superpixel_mean_Features, HyperCube, superpix_img0, superpix_img,
                'CORR', conn_pattern, dist_mode2, center_mode2
            )
            S_W_tmp = WeightMatNormalization(WeightMat)
        #
        if S_W.nnz==0:
            S_W = csr_matrix(S_W_tmp)
            nodes_to_add = nodes_to_add_tmp
            S_W_no_normalize = WeightMat
        else:
            if superpix_num == superpix_allnum[-1]:
                S_W, S_W_no_normalize = MultiGraph_Fusion_PCA_sub(S_W_tmp, S_W)
            else:
                S_W, S_W_no_normalize = MultiGraph_Fusion_PCA_sub(S_W_tmp, S_W)
            # nodes_to_add = nodes_to_add | nodes_to_add_tmp
            nodes_to_add = nodes_to_add.maximum(nodes_to_add_tmp)

    # Step 3: Incorporate training sample locations and labels
    for train_i in range(len(TrainSamLoc)):
        itrainloc = TrainSamLoc[train_i]
        sp0_idx1 = superpix_img0[itrainloc]-1
        itrainlabel = TrainLabels[train_i]
        otherloc = TrainSamLoc[TrainLabels == itrainlabel]
        sp0_idx2 = superpix_img0[otherloc]-1
        sp0_idx2 = np.unique(sp0_idx2)

        S_W_no_normalize[sp0_idx1, sp0_idx2] = 1
        S_W_no_normalize[sp0_idx2, sp0_idx1] = 1
        S_W_no_normalize[sp0_idx1, sp0_idx1] = 0

    return S_W_no_normalize, superpix_img0, Superpixel_mean_Features


def FSPG_graph_construction_bigdata(HyperCube, superpix_num0, superpix_allnum, conn_pattern,
                                    TrainSamLoc, TrainLabels, dist_mode2, center_mode2
                                    ):
    Height, Width, Bands = HyperCube.shape
    HyperCube_vec = HyperCube.reshape(Width * Height, Bands)  # 3D to 2D
    S_W = csr_matrix([])

    # Step 1: Superpixel segmentation for superpix_num0
    superpix_img0 = superpixel_cut(HyperCube, superpix_num0)
    sp_num = np.max(superpix_img0)

    for select_i in range(1, sp_num + 1):
        pos_idx = np.where(superpix_img0 == select_i)[0]
        spTrainflag = np.isin(TrainSamLoc, pos_idx)
        spTrainSamLoc = TrainSamLoc[spTrainflag]
        spTrainLabels = TrainLabels[spTrainflag]

        if len(spTrainSamLoc) > 0:
            spTrainLabels = yl_label_reranging(spTrainLabels) + 1
            if np.max(spTrainLabels) > 1:
                # Compute weight matrix for superpixels
                WeightMat_sp = SNG_sp_patch(HyperCube_vec[pos_idx, :], pos_idx, Height, Width)
                N_sp = WeightMat_sp.shape[0]
                d = np.sum(WeightMat_sp, axis=1)
                d[d != 0] = 1 / np.sqrt(d[d != 0])
                Dinv_sp = spdiags(d, 0, N_sp, N_sp)
                SW_sp = Dinv_sp @ WeightMat_sp @ Dinv_sp
                initial_Labels_sp = np.zeros((N_sp, np.max(spTrainLabels)))

                for spTr_i in range(len(spTrainSamLoc)):
                    tmp_spTrainSamLoc = spTrainSamLoc[spTr_i]
                    pos_idx_fortmpsp = np.where(pos_idx == tmp_spTrainSamLoc)[0]
                    initial_Labels_sp[pos_idx_fortmpsp, spTrainLabels[spTr_i] - 1] = 1

                initial_Labels_sp = csr_matrix(initial_Labels_sp)
                para_alpha = 0.5
                UU = eye(N_sp) - para_alpha * SW_sp
                F = inv(UU) @ initial_Labels_sp  # F = (I - alpha * S)^(-1) * Y
                PredLabels_sp = np.argmax(F.toarray(), axis=1)

                # Update superpixel labels based on predictions
                for splabel_i in range(1, np.max(PredLabels_sp) + 1):
                    sploc_i = np.where(PredLabels_sp == splabel_i)[0]
                    max_sp_ind = np.max(superpix_img0)
                    superpix_img0[pos_idx[sploc_i]] = max_sp_ind + 1

    superpix_img0 = yl_label_reranging(superpix_img0) + 1

    # Compute Superpixel Mean Features
    sp_num0 = np.max(superpix_img0)
    ALL_data = HyperCube.reshape(Width * Height, Bands)  # 3D to 2D
    Superpixel_mean_Features = []

    for i in range(0, sp_num0):
        sp_idx1 = np.where(superpix_img0 == i+1)[0]
        X1 = np.mean(ALL_data[sp_idx1, :], axis=0)
        Superpixel_mean_Features.append(X1)

    Superpixel_mean_Features = np.array(Superpixel_mean_Features)

    # Step 2: Iterate through different superpixel numbers
    for superpix_num in superpix_allnum:
        # if superpix_num==900:
        #     superpix_img = superpix_img01
        # if superpix_num==1800:
        #     superpix_img = superpix_img02

        # Superpixel segmentation
        superpix_img = superpixel_cut(HyperCube, superpix_num)
        sp_num = np.max(superpix_img)

        for select_i in range(1, sp_num + 1):
            pos_idx = np.where(superpix_img == select_i)[0]
            spTrainflag = np.isin(TrainSamLoc, pos_idx)
            spTrainSamLoc = TrainSamLoc[spTrainflag]
            spTrainLabels = TrainLabels[spTrainflag]

            if len(spTrainSamLoc) > 0:
                spTrainLabels = yl_label_reranging(spTrainLabels) + 1
                if np.max(spTrainLabels) > 1:
                    # Compute weight matrix for superpixels
                    WeightMat_sp = SNG_sp_patch(HyperCube_vec[pos_idx, :], pos_idx, Height, Width)
                    N_sp = WeightMat_sp.shape[0]
                    # d = np.sum(WeightMat_sp, axis=1)
                    d = np.array(WeightMat_sp.sum(axis=1)).flatten()  # 计算每行的和 (需要转换为数组)
                    # 创建 Dinv 对角矩阵，使用 spdiags
                    d[d != 0] = 1 / np.sqrt(d[d != 0])
                    Dinv_sp = spdiags(d, 0, N_sp, N_sp)
                    SW_sp = Dinv_sp @ WeightMat_sp @ Dinv_sp
                    initial_Labels_sp = np.zeros((N_sp, np.max(spTrainLabels)))

                    for spTr_i in range(len(spTrainSamLoc)):
                        tmp_spTrainSamLoc = spTrainSamLoc[spTr_i]
                        pos_idx_fortmpsp = np.where(pos_idx == tmp_spTrainSamLoc)[0]
                        initial_Labels_sp[pos_idx_fortmpsp, spTrainLabels[spTr_i] - 1] = 1

                    initial_Labels_sp = csr_matrix(initial_Labels_sp)
                    para_alpha = 0.5
                    UU = eye(N_sp) - para_alpha * SW_sp
                    # other data
                    F = inv(UU) @ initial_Labels_sp  # F = (I - alpha * S)^(-1) * Y
                    # F = sp.linalg.spsolve(UU, initial_Labels_sp)
                    PredLabels_sp = np.argmax(F.toarray(), axis=1)

                    # Update superpixel labels based on predictions
                    for splabel_i in range(1, np.max(PredLabels_sp) + 1):
                        sploc_i = np.where(PredLabels_sp == splabel_i)[0]
                        max_sp_ind = np.max(superpix_img)
                        superpix_img[pos_idx[sploc_i]] = max_sp_ind + 1

        superpix_img = yl_label_reranging(superpix_img) + 1

        # Compute weight matrix based on superpixel number
        if superpix_num > 11:
            WeightMat, nodes_to_add_tmp = SuperPixWeightMat_Experiment_with_Block2Block_bigdata(
                Superpixel_mean_Features, HyperCube, superpix_img0, superpix_img,
                'CORR', conn_pattern, dist_mode2, center_mode2
            )
            S_W_tmp = WeightMatNormalization(WeightMat)


        if S_W.nnz==0:
            S_W = csr_matrix(S_W_tmp)
            nodes_to_add = nodes_to_add_tmp
            S_W_no_normalize = WeightMat
        else:
            if superpix_num == superpix_allnum[-1]:
                S_W, S_W_no_normalize = MultiGraph_Fusion_PCA_sub(S_W_tmp, S_W)
            else:
                S_W, S_W_no_normalize = MultiGraph_Fusion_PCA_sub(S_W_tmp, S_W)
            # nodes_to_add = nodes_to_add | nodes_to_add_tmp
            nodes_to_add = nodes_to_add.maximum(nodes_to_add_tmp)

    # Step 3: Incorporate training sample locations and labels
    for train_i in range(len(TrainSamLoc)):
        itrainloc = TrainSamLoc[train_i]
        sp0_idx1 = superpix_img0[itrainloc]-1
        itrainlabel = TrainLabels[train_i]
        otherloc = TrainSamLoc[TrainLabels == itrainlabel]
        sp0_idx2 = superpix_img0[otherloc]-1
        sp0_idx2 = np.unique(sp0_idx2)

        S_W_no_normalize[sp0_idx1, sp0_idx2] = 1
        S_W_no_normalize[sp0_idx2, sp0_idx1] = 1
        S_W_no_normalize[sp0_idx1, sp0_idx1] = 0

    return S_W_no_normalize, superpix_img0, Superpixel_mean_Features


def FSPG_graph_constructionV2(HyperCube, superpix_allnum, conn_pattern, TrainSamLoc, TrainLabels,
                              superpix_img01, superpix_img02, dist_mode1, center_mode1):
    Height, Width, Bands = HyperCube.shape
    HyperCube_vec = HyperCube.reshape(Width * Height, Bands)  # 3D to 2D
    S_W = csr_matrix([])
    fuse_idx = 0

    for superpix_num in superpix_allnum:
        # Superpixel segmentation
        if superpix_num==300:
            superpix_img = superpix_img01
        if superpix_num==600:
            superpix_img = superpix_img02
        # Compute weight matrix for superpixels
        if superpix_num > 11:
            WeightMat, nodes_to_add_tmp = SuperPixWeightMat_Experiment_with_Block2Block(
                HyperCube, superpix_img, 'CORR', conn_pattern, dist_mode1, center_mode1
            )
            S_W_tmp = WeightMatNormalization(WeightMat)

        if S_W.nnz==0:
            S_W = csr_matrix(S_W_tmp)
            nodes_to_add = nodes_to_add_tmp
            S_W_no_normalize = WeightMat
        else:
            if superpix_num == superpix_allnum[-1]:

                S_W, S_W_no_normalize = MultiGraph_Fusion_PCA_sub(S_W_tmp, S_W)
            else:
                S_W, S_W_no_normalize = MultiGraph_Fusion_PCA_sub(S_W_tmp, S_W)
            # nodes_to_add = nodes_to_add | nodes_to_add_tmp
            nodes_to_add = nodes_to_add.maximum(nodes_to_add_tmp)

    # Incorporate training sample locations and labels
    for train_i in range(len(TrainSamLoc)):
        itrainloc = TrainSamLoc[train_i]
        itrainlabel = TrainLabels[train_i]
        otherloc = TrainSamLoc[TrainLabels == itrainlabel]

        S_W_no_normalize[itrainloc, otherloc] = 1
        S_W_no_normalize[otherloc, itrainloc] = 1
        S_W_no_normalize[itrainloc, itrainloc] = 0

    return S_W_no_normalize

def FSPG_graph_construction_bigdataV2(HyperCube, superpix_num0, superpix_allnum, conn_pattern, TrainSamLoc, TrainLabels
                                    ,superpix_img0,superpix_img01,superpix_img02, dist_mode2, center_mode2):
    Height, Width, Bands = HyperCube.shape
    HyperCube_vec = HyperCube.reshape(Width * Height, Bands)  # 3D to 2D
    S_W = csr_matrix([])

    # Step 1: Superpixel segmentation for superpix_num0
    # superpix_img0 = superpixel_cut(HyperCube, superpix_num0)
    # sp_num = np.max(superpix_img0)
    #
    # for select_i in range(1, sp_num + 1):
    #     pos_idx = np.where(superpix_img0 == select_i)[0]
    #     spTrainflag = np.isin(TrainSamLoc, pos_idx)
    #     spTrainSamLoc = TrainSamLoc[spTrainflag]
    #     spTrainLabels = TrainLabels[spTrainflag]
    #
    #     if len(spTrainSamLoc) > 0:
    #         spTrainLabels = yl_label_reranging(spTrainLabels) + 1
    #         if np.max(spTrainLabels) > 1:
    #             # Compute weight matrix for superpixels
    #             WeightMat_sp = SNG_sp_patch(HyperCube_vec[pos_idx, :], pos_idx, Height, Width)
    #             N_sp = WeightMat_sp.shape[0]
    #             d = np.sum(WeightMat_sp, axis=1)
    #             d[d != 0] = 1 / np.sqrt(d[d != 0])
    #             Dinv_sp = spdiags(d, 0, N_sp, N_sp)
    #             SW_sp = Dinv_sp @ WeightMat_sp @ Dinv_sp
    #             initial_Labels_sp = np.zeros((N_sp, np.max(spTrainLabels)))
    #
    #             for spTr_i in range(len(spTrainSamLoc)):
    #                 tmp_spTrainSamLoc = spTrainSamLoc[spTr_i]
    #                 pos_idx_fortmpsp = np.where(pos_idx == tmp_spTrainSamLoc)[0]
    #                 initial_Labels_sp[pos_idx_fortmpsp, spTrainLabels[spTr_i] - 1] = 1
    #
    #             initial_Labels_sp = csr_matrix(initial_Labels_sp)
    #             para_alpha = 0.5
    #             UU = eye(N_sp) - para_alpha * SW_sp
    #             F = inv(UU) @ initial_Labels_sp  # F = (I - alpha * S)^(-1) * Y
    #             PredLabels_sp = np.argmax(F.toarray(), axis=1)
    #
    #             # Update superpixel labels based on predictions
    #             for splabel_i in range(1, np.max(PredLabels_sp) + 1):
    #                 sploc_i = np.where(PredLabels_sp == splabel_i)[0]
    #                 max_sp_ind = np.max(superpix_img0)
    #                 superpix_img0[pos_idx[sploc_i]] = max_sp_ind + 1
    #
    # superpix_img0 = yl_label_reranging(superpix_img0) + 1

    # Compute Superpixel Mean Features
    sp_num0 = np.max(superpix_img0)
    ALL_data = HyperCube.reshape(Width * Height, Bands)  # 3D to 2D
    Superpixel_mean_Features = []

    for i in range(0, sp_num0):
        sp_idx1 = np.where(superpix_img0 == i+1)[0]
        X1 = np.mean(ALL_data[sp_idx1, :], axis=0)
        Superpixel_mean_Features.append(X1)

    Superpixel_mean_Features = np.array(Superpixel_mean_Features)

    # Step 2: Iterate through different superpixel numbers
    for superpix_num in superpix_allnum:
        if superpix_num==300:
            superpix_img = superpix_img01
        if superpix_num==600:
            superpix_img = superpix_img02
        # Superpixel segmentation
        # superpix_img = superpixel_cut(HyperCube, superpix_num)
        # sp_num = np.max(superpix_img)
        #
        # for select_i in range(1, sp_num + 1):
        #     pos_idx = np.where(superpix_img == select_i)[0]
        #     spTrainflag = np.isin(TrainSamLoc, pos_idx)
        #     spTrainSamLoc = TrainSamLoc[spTrainflag]
        #     spTrainLabels = TrainLabels[spTrainflag]
        #
        #     if len(spTrainSamLoc) > 0:
        #         spTrainLabels = yl_label_reranging(spTrainLabels) + 1
        #         if np.max(spTrainLabels) > 1:
        #             # Compute weight matrix for superpixels
        #             WeightMat_sp = SNG_sp_patch(HyperCube_vec[pos_idx, :], pos_idx, Height, Width)
        #             N_sp = WeightMat_sp.shape[0]
        #             d = np.sum(WeightMat_sp, axis=1)
        #             d[d != 0] = 1 / np.sqrt(d[d != 0])
        #             Dinv_sp = spdiags(d, 0, N_sp, N_sp)
        #             SW_sp = Dinv_sp @ WeightMat_sp @ Dinv_sp
        #             initial_Labels_sp = np.zeros((N_sp, np.max(spTrainLabels)))
        #
        #             for spTr_i in range(len(spTrainSamLoc)):
        #                 tmp_spTrainSamLoc = spTrainSamLoc[spTr_i]
        #                 pos_idx_fortmpsp = np.where(pos_idx == tmp_spTrainSamLoc)[0]
        #                 initial_Labels_sp[pos_idx_fortmpsp, spTrainLabels[spTr_i] - 1] = 1
        #
        #             initial_Labels_sp = csr_matrix(initial_Labels_sp)
        #             para_alpha = 0.5
        #             UU = eye(N_sp) - para_alpha * SW_sp
        #             F = inv(UU) @ initial_Labels_sp  # F = (I - alpha * S)^(-1) * Y
        #             PredLabels_sp = np.argmax(F.toarray(), axis=1)
        #
        #             # Update superpixel labels based on predictions
        #             for splabel_i in range(1, np.max(PredLabels_sp) + 1):
        #                 sploc_i = np.where(PredLabels_sp == splabel_i)[0]
        #                 max_sp_ind = np.max(superpix_img)
        #                 superpix_img[pos_idx[sploc_i]] = max_sp_ind + 1
        #
        # superpix_img = yl_label_reranging(superpix_img) + 1

        # Compute weight matrix based on superpixel number
        if superpix_num > 11:
            WeightMat, nodes_to_add_tmp = SuperPixWeightMat_Experiment_with_Block2Block_bigdata(
                Superpixel_mean_Features, HyperCube, superpix_img0, superpix_img,
                'CORR', conn_pattern, dist_mode2, center_mode2
            )
            S_W_tmp = WeightMatNormalization(WeightMat)


        if S_W.nnz==0:
            S_W = csr_matrix(S_W_tmp)
            nodes_to_add = nodes_to_add_tmp
            S_W_no_normalize = WeightMat
        else:
            if superpix_num == superpix_allnum[-1]:
                S_W, S_W_no_normalize = MultiGraph_Fusion_PCA_sub(S_W_tmp, S_W)
            else:
                S_W, S_W_no_normalize = MultiGraph_Fusion_PCA_sub(S_W_tmp, S_W)
            # nodes_to_add = nodes_to_add | nodes_to_add_tmp
            nodes_to_add = nodes_to_add.maximum(nodes_to_add_tmp)

    # Step 3: Incorporate training sample locations and labels
    for train_i in range(len(TrainSamLoc)):
        itrainloc = TrainSamLoc[train_i]
        sp0_idx1 = superpix_img0[itrainloc]-1
        itrainlabel = TrainLabels[train_i]
        otherloc = TrainSamLoc[TrainLabels == itrainlabel]
        sp0_idx2 = superpix_img0[otherloc]-1
        sp0_idx2 = np.unique(sp0_idx2)

        S_W_no_normalize[sp0_idx1, sp0_idx2] = 1
        S_W_no_normalize[sp0_idx2, sp0_idx1] = 1
        S_W_no_normalize[sp0_idx1, sp0_idx1] = 0

    return S_W_no_normalize, superpix_img0, Superpixel_mean_Features

