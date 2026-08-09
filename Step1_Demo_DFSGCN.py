import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random
from matplotlib import cm
from sklearn import metrics
import time
from sklearn import preprocessing
import torch
import DFSGCN_model
import scipy.io as scio
import os
from utils import *
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
samples_type = ['ratio', 'same_num'][1]

for (FLAG, curr_train_ratio) in [(2, 5)]:
    torch.cuda.empty_cache()
    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    AVG_ALL = []
    F1score_ALL = []
    OA_med_ALL=[]
    AA_med_ALL = []
    kappa_med_ALL = []
    F1score_med_ALL = []
    AVG_med_ALL = []
    F1perclass_med_ALL = []
    Train_Time_ALL = []
    Test_Time_ALL = []

    Seed_List = [1, 2, 3, 4, 5]  # 随机种子点

    if FLAG == 1:
        data_mat = sio.loadmat('../Demo_FSPG_v2\\Houston2018\\DFC2018_20.mat')
        data = data_mat['DFC2018']
        gt_mat = sio.loadmat('../Demo_FSPG_v2\\Houston2018\\DFC2018_gt20.mat')
        gt = gt_mat['DFC2018_gt']
        class_count = 20  # 样本类别数
        # 参数预设
        # train_ratio = 0.05  # 训练集比例。注意，训练集为按照‘每类’随机选取
        val_ratio = 0.05  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
        learning_rate = 5e-4  # 学习率
        max_epoch = 400  # 迭代次数
        dataset_name = "Houston2018_"  # 数据集名称
        data_name = "Houston2018"
        # superpixel_scale = 200
        HyperCube = fmapminmax(data.astype(np.float64))
        conn_pattern = 'combine'
        Height, Width, _ = HyperCube.shape  # Assuming HyperCube is a NumPy array
        superpix_num0 = int(np.floor(Height * Width / 300))
        superpix_num2 = int(np.floor(Height * Width / 400))
        superpix_num3 = int(np.floor(superpix_num2 / 2.5))
        superpix_allnum=[600,1200]
        superpix_num0 = int(np.floor(Height * Width / 80))
        superpix_num2 = int(np.floor(Height * Width / 240))
        superpix_num3 = int(np.floor(superpix_num2 / 2.5))
        # superpix_allnum=[2400,4800]
        superpix_allnum=[20000,30000]
        kernel_size = 9
        dist_mode2, center_mode2=0,0 #

    if FLAG == 2:
        data_mat = sio.loadmat('HyperImage_data\\LapYuHeSiDi\\LapYuHeSiDi.mat')
        data = data_mat['LapYuHeSiDi']
        gt_mat = sio.loadmat('HyperImage_data\\LapYuHeSiDi\\LapYuHeSiDi_label.mat')
        gt = gt_mat['LapYuHeSiDi_label']

        # 参数预设
        # train_ratio = 0.05  # 训练集比例。注意，训练集为按照‘每类’随机选取
        val_ratio = 0.05  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
        class_count = 8  # 样本类别数
        learning_rate = 5e-4  # 学习率
        max_epoch = 600  # 迭代次数
        dataset_name = "LapYuHeSiDi_"  # 数据集名称
        data_name = "LapYuHeSiDi"
        # superpixel_scale = 200
        HyperCube = fmapminmax(data.astype(np.float64))
        conn_pattern = 'combine'
        Height, Width, _ = HyperCube.shape  # Assuming HyperCube is a NumPy array
        superpix_num0 = int(np.floor(Height * Width / 50))
        superpix_num2 = int(np.floor(Height * Width / 40))
        superpix_num3 = int(np.floor(superpix_num2 / 2.5))
        superpix_allnum=[600,1200]
        kernel_size = 9
        dist_mode2, center_mode2=0,0
        pass
    if FLAG == 3:
        data_mat = sio.loadmat('HyperImage_data\\WHU_Hi_HanChuan\\WHU_Hi_HanChuan.mat')
        data = data_mat['WHU_Hi_HanChuan']
        gt_mat = sio.loadmat('HyperImage_data\\WHU_Hi_HanChuan\\WHU_Hi_HanChuan_gt.mat')
        gt = gt_mat['WHU_Hi_HanChuan_gt']

        # 参数预设
        # train_ratio = 0.05  # 训练集比例。注意，训练集为按照‘每类’随机选取
        val_ratio = 0.05  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
        class_count = 16  # 样本类别数
        learning_rate = 5e-4  # 学习率
        max_epoch = 600  # 迭代次数
        dataset_name = "WHU_Hi_HanChuan_"  # 数据集名称
        data_name = "WHU_Hi_HanChuan"
        # superpixel_scale = 200

        HyperCube = fmapminmax(data.astype(np.float64))
        conn_pattern = 'combine'
        Height, Width, _ = HyperCube.shape  # Assuming HyperCube is a NumPy array
        superpix_num0 = int(np.floor(Height * Width / 100))
        superpix_num2 = int(np.floor(Height * Width / 60))
        superpix_num3 = int(np.floor(superpix_num2 / 2.5))
        superpix_allnum = [900, 1800]
        kernel_size = 9
        dist_mode2, center_mode2=0,1
        pass
    if FLAG == 4:
        data_mat = sio.loadmat('HyperImage_data\\IndianaPine\\IndianaPine.mat')
        data = data_mat['IndianaPine']
        gt_mat = sio.loadmat('HyperImage_data\\IndianaPine\\IndianaPine_gt.mat')
        gt = gt_mat['IndianaPine_gt']

        val_ratio = 0.05
        class_count = 16
        learning_rate = 8e-4
        # max_epoch = 600
        max_epoch = 300
        dataset_name = "IndianaPine_"
        data_name = "IndianPines"
        # HyperCube = fmapminmax(data.astype(np.float64))
        HyperCube =data.copy()
        conn_pattern = 'combine'
        Height, Width, _ = HyperCube.shape  # Assuming HyperCube is a NumPy array
        superpix_num0 = int(np.floor(Height * Width / 1))
        superpix_num2 = int(np.floor(Height * Width / 4))
        superpix_num3 = int(np.floor(superpix_num2 / 2.5))
        superpix_allnum=[300,600]
        dist_mode2, center_mode2=0,2
        # dist_mode2, center_mode2=0,0
        kernel_size = 7
        pass
    if FLAG == 5:
        data_mat = sio.loadmat('HyperImage_data\\Zaoyuan\\Zaoyuan.mat')
        data = data_mat['Zaoyuan']
        gt_mat = sio.loadmat('HyperImage_data\\Zaoyuan\\Zaoyuan_gt.mat')
        gt = gt_mat['Zaoyuan_gt']

        val_ratio = 0.05
        class_count = 8
        learning_rate = 5e-4
        max_epoch = 300
        dataset_name = "Zaoyuan_"
        data_name = "Zaoyuan2"
        Seed_List = [4, 5, 8, 13, 20]  # 随机种子点
        # HyperCube = fmapminmax(data.astype(np.float64))
        HyperCube =data.copy()
        conn_pattern = 'combine'
        Height, Width, _ = HyperCube.shape  # Assuming HyperCube is a NumPy array
        superpix_num0 = int(np.floor(Height * Width / 1))
        superpix_num2 = int(np.floor(Height * Width / 4))
        superpix_num3 = int(np.floor(superpix_num2 / 2.5))
        superpix_allnum=[900,1500]
        kernel_size = 11
        dist_mode2, center_mode2=0,1
        pass
    if FLAG == 6:
        data_mat = sio.loadmat('HyperImage_data\\zhengzhou\\zhengzhou_20191110_data.mat')
        data = data_mat['zhengzhou_20191110_data']
        gt_mat = sio.loadmat('HyperImage_data\\zhengzhou\\zhengzhou_20191110_label.mat')
        gt = gt_mat['zhengzhou_20191110_label']

        # 参数预设
        # train_ratio = 0.05  # 训练集比例。注意，训练集为按照‘每类’随机选取
        val_ratio = 0.05  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
        class_count = 4  # 样本类别数
        learning_rate = 5e-4  # 学习率
        max_epoch = 600  # 迭代次数
        dataset_name = "zhengzhou_20191110_"  # 数据集名称
        data_name = "zhengzhou_20191110"
        pass
    if FLAG == 7:
        data_mat = sio.loadmat('HyperImage_data\\honghu\\honghu.mat')
        data = data_mat['honghu']
        gt_mat = sio.loadmat('HyperImage_data\\honghu\\honghu_gt_sup.mat')
        gt = gt_mat['honghu_gt_sup']

        val_ratio = 0.05
        class_count = 5
        learning_rate = 5e-4
        max_epoch = 600
        dataset_name = "honghu_"
        data_name = "honghu"
        pass
    if FLAG == 8:
        data_mat = sio.loadmat('..\\HyperImage_data\\paviaU\\PaviaU.mat')
        data = data_mat['paviaU']
        gt_mat = sio.loadmat('..\\HyperImage_data\\paviaU\\Pavia_University_gt.mat')
        gt = gt_mat['pavia_university_gt']

        # 参数预设
        # train_ratio = 0.01  # 训练集比例。注意，训练集为按照‘每类’随机选取
        val_ratio = 0.05  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
        class_count = 9  # 样本类别数
        learning_rate = 5e-4  # 学习率
        max_epoch = 600  # 迭代次数
        dataset_name = "PaviaU_"  # 数据集名称
        data_name = "PaviaU"

        HyperCube = fmapminmax(data.astype(np.float64))
        conn_pattern = 'combine'
        Height, Width, _ = HyperCube.shape  # Assuming HyperCube is a NumPy array
        superpix_num0 = int(np.floor(Height * Width / 100))
        superpix_num2 = int(np.floor(Height * Width / 60))
        superpix_num3 = int(np.floor(superpix_num2 / 2.5))
        superpix_allnum = [900, 1800]
        kernel_size = 9
        pass
    if FLAG == 9:
        data_mat = sio.loadmat('..\\HyperImage_data\\KSC\\KSC.mat')
        data = data_mat['KSC']
        gt_mat = sio.loadmat('..\\HyperImage_data\\KSC\\KSC_gt.mat')
        gt = gt_mat['KSC_gt']

        # 参数预设
        # train_ratio = 0.05  # 训练集比例。注意，训练集为按照‘每类’随机选取
        val_ratio = 0.05  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
        class_count = 13  # 样本类别数
        learning_rate = 5e-4  # 学习率
        max_epoch = 600  # 迭代次数
        dataset_name = "KSC_"  # 数据集名称
        pass

    ###########
    train_samples_per_class = curr_train_ratio  # 当定义为每类样本个数时,则该参数更改为训练样本数
    val_samples = class_count
    train_ratio = curr_train_ratio
    cmap = cm.get_cmap('jet', class_count + 1)
    plt.set_cmap(cmap)
    m, n, d = data.shape  # 高光谱数据的三个维度

    # 数据standardization标准化,即提前全局BN
    orig_data = data
    height, width, bands = data.shape  # 原始高光谱数据的三个维度
    data = np.reshape(data, [height * width, bands])
    minMax = preprocessing.StandardScaler()
    data = minMax.fit_transform(data)
    data = np.reshape(data, [height, width, bands])
    del orig_data

    # # 打印每类样本个数
    # gt_reshape=np.reshape(gt, [-1])
    # for i in range(class_count):
    #     idx = np.where(gt_reshape == i + 1)[-1]
    #     samplesCount = len(idx)
    #     print(samplesCount)

    for curr_seed in Seed_List:
        # step1: 随机10%数据作为训练样本。方式：给出训练数据与测试数据的GT
        # random.seed(curr_seed)
        # gt_reshape = np.reshape(gt, [-1])
        # train_rand_idx = []
        # val_rand_idx = []
        # if samples_type == 'ratio':
        #     for i in range(class_count):
        #         idx = np.where(gt_reshape == i + 1)[-1]
        #         samplesCount = len(idx)
        #         rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
        #         rand_idx = random.sample(rand_list,
        #                                  np.ceil(samplesCount * train_ratio).astype('int32'))  # 随机数数量 四舍五入(改为上取整)
        #         rand_real_idx_per_class = idx[rand_idx]
        #         train_rand_idx.append(rand_real_idx_per_class)
        #     train_rand_idx = np.array(train_rand_idx)
        #     train_data_index = []
        #     for c in range(train_rand_idx.shape[0]):
        #         a = train_rand_idx[c]
        #         for j in range(a.shape[0]):
        #             train_data_index.append(a[j])
        #     train_data_index = np.array(train_data_index)
        #
        #     ##将测试集（所有样本，包括训练样本）也转化为特定形式
        #     train_data_index = set(train_data_index)
        #     all_data_index = [i for i in range(len(gt_reshape))]
        #     all_data_index = set(all_data_index)
        #
        #     # 背景像元的标签
        #     background_idx = np.where(gt_reshape == 0)[-1]
        #     background_idx = set(background_idx)
        #     test_data_index = all_data_index - train_data_index - background_idx
        #
        #     # 从测试集中随机选取部分样本作为验证集
        #     val_data_count = int(val_ratio * (len(test_data_index) + len(train_data_index)))  # 验证集数量
        #     val_data_index = random.sample(test_data_index, val_data_count)
        #     val_data_index = set(val_data_index)
        #     test_data_index = test_data_index - val_data_index  # 由于验证集为从测试集分裂出，所以测试集应减去验证集
        #
        #     # 将训练集 验证集 测试集 整理
        #     test_data_index = list(test_data_index)
        #     train_data_index = list(train_data_index)
        #     val_data_index = list(val_data_index)
        #
        # if samples_type == 'same_num':
        #     for i in range(class_count):
        #         idx = np.where(gt_reshape == i + 1)[-1]
        #         samplesCount = len(idx)
        #         real_train_samples_per_class = train_samples_per_class
        #         rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
        #         if real_train_samples_per_class > samplesCount:
        #             real_train_samples_per_class = samplesCount
        #         rand_idx = random.sample(rand_list,
        #                                  real_train_samples_per_class)  # 随机数数量 四舍五入(改为上取整)
        #         rand_real_idx_per_class_train = idx[rand_idx[0:real_train_samples_per_class]]
        #         train_rand_idx.append(rand_real_idx_per_class_train)
        #     train_rand_idx = np.array(train_rand_idx)
        #     val_rand_idx = np.array(val_rand_idx)
        #     train_data_index = []
        #     for c in range(train_rand_idx.shape[0]):
        #         a = train_rand_idx[c]
        #         for j in range(a.shape[0]):
        #             train_data_index.append(a[j])
        #     train_data_index = np.array(train_data_index)
        #
        #     train_data_index = set(train_data_index)
        #     all_data_index = [i for i in range(len(gt_reshape))]
        #     all_data_index = set(all_data_index)
        #
        #     # 背景像元的标签
        #     background_idx = np.where(gt_reshape == 0)[-1]
        #     background_idx = set(background_idx)
        #     test_data_index = all_data_index - train_data_index - background_idx
        #
        #     # 从测试集中随机选取部分样本作为验证集
        #     val_data_count = int(val_samples)  # 验证集数量
        #     val_data_index = random.sample(test_data_index, val_data_count)
        #     val_data_index = set(val_data_index)
        #
        #     test_data_index = test_data_index - val_data_index
        #     # 将训练集 验证集 测试集 整理
        #     test_data_index = list(test_data_index)
        #     train_data_index = list(train_data_index)
        #     val_data_index = list(val_data_index)
        #
        # # 获取训练样本的标签图
        # train_samples_gt = np.zeros(gt_reshape.shape)
        # for i in range(len(train_data_index)):
        #     train_samples_gt[train_data_index[i]] = gt_reshape[train_data_index[i]]
        #     pass
        #
        # # 获取测试样本的标签图
        # test_samples_gt = np.zeros(gt_reshape.shape)
        # for i in range(len(test_data_index)):
        #     test_samples_gt[test_data_index[i]] = gt_reshape[test_data_index[i]]
        #     pass
        #
        # # 获取验证集样本的标签图
        # val_samples_gt = np.zeros(gt_reshape.shape)
        # for i in range(len(val_data_index)):
        #     val_samples_gt[val_data_index[i]] = gt_reshape[val_data_index[i]]
        #     pass
        #
        # train_samples_gt=np.reshape(train_samples_gt,[height,width])
        # test_samples_gt=np.reshape(test_samples_gt,[height,width])
        # val_samples_gt=np.reshape(val_samples_gt,[height,width])

        data_labels_mats = scio.loadmat(
            'trainTestSplit/{}/sample{}_run{}.mat'.format(data_name, curr_train_ratio, curr_seed))
        train_samples_gt, test_samples_gt = np.array(data_labels_mats['train_gt']), np.array(
            data_labels_mats['test_gt'])
        # val_samples_gt = extract_validation_samples(test_samples_gt, val_ratio, random_seed=42)
        val_samples_gt = test_samples_gt.copy()

        Test_GT = np.reshape(test_samples_gt, [m, n])  # 测试样本图

        train_samples_gt_onehot = GT_To_One_Hot(train_samples_gt, class_count)
        test_samples_gt_onehot = GT_To_One_Hot(test_samples_gt, class_count)
        val_samples_gt_onehot = GT_To_One_Hot(val_samples_gt, class_count)

        train_samples_gt_onehot = np.reshape(train_samples_gt_onehot, [-1, class_count]).astype(int)
        test_samples_gt_onehot = np.reshape(test_samples_gt_onehot, [-1, class_count]).astype(int)
        val_samples_gt_onehot = np.reshape(val_samples_gt_onehot, [-1, class_count]).astype(int)

        train_gt_flat = train_samples_gt.flatten()
        TrainSamLoc = np.where(train_gt_flat != 0)[0]  # Indices of non-zero elements (1D)
        TrainLabels = train_gt_flat[TrainSamLoc]
        test_gt_flat = test_samples_gt.flatten()
        TestSamLoc = np.where(test_gt_flat != 0)[0]  # Indices of non-zero elements (1D)
        TestLabels = test_gt_flat[TestSamLoc]
        # Call the FSPG_graph_construction_bigdata function
        _, initial_Labels1, superpix_img1,\
            sp_num1, Superpixel_mean_Features1, WeightMat1 = DFSG_gen(HyperCube, superpix_num0,
                                                                      superpix_allnum, conn_pattern,
                                                                      TrainSamLoc, TrainLabels,
                                                                      class_count, dist_mode2, center_mode2)

        _, initial_Labels2, superpix_img2,\
            sp_num2, Superpixel_mean_Features2, WeightMat2 = DFSG_gen(HyperCube, superpix_num2,
                                                                      superpix_allnum, conn_pattern,
                                                                      TrainSamLoc, TrainLabels,
                                                                      class_count, dist_mode2, center_mode2)

        ############制作训练数据和测试数据的gt掩膜.根据GT将带有标签的像元设置为全1向量##############
        # 训练集
        train_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        train_samples_gt = np.reshape(train_samples_gt, [m * n])
        for i in range(m * n):
            if train_samples_gt[i] != 0:
                train_label_mask[i] = temp_ones
        train_label_mask = np.reshape(train_label_mask, [m * n, class_count])

        # 测试集
        test_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        test_samples_gt = np.reshape(test_samples_gt, [m * n])
        for i in range(m * n):
            if test_samples_gt[i] != 0:
                test_label_mask[i] = temp_ones
        test_label_mask = np.reshape(test_label_mask, [m * n, class_count])

        # 验证集
        val_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        val_samples_gt = np.reshape(val_samples_gt, [m * n])
        for i in range(m * n):
            if val_samples_gt[i] != 0:
                val_label_mask[i] = temp_ones
        val_label_mask = np.reshape(val_label_mask, [m * n, class_count])

        # torch.manual_seed(3407)
        if data_name != "LapYuHeSiDi":
            torch.manual_seed(3408)  ## IndianP
        # 转到GPU
        train_samples_gt = torch.from_numpy(train_samples_gt.astype(np.float32)).to(device)
        test_samples_gt = torch.from_numpy(test_samples_gt.astype(np.float32)).to(device)
        val_samples_gt = torch.from_numpy(val_samples_gt.astype(np.float32)).to(device)
        # 转到GPU
        train_samples_gt_onehot = torch.from_numpy(train_samples_gt_onehot.astype(np.float32)).to(device)
        test_samples_gt_onehot = torch.from_numpy(test_samples_gt_onehot.astype(np.float32)).to(device)
        val_samples_gt_onehot = torch.from_numpy(val_samples_gt_onehot.astype(np.float32)).to(device)
        # 转到GPU
        train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(device)
        test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(device)
        val_label_mask = torch.from_numpy(val_label_mask.astype(np.float32)).to(device)

        del Superpixel_mean_Features1,Superpixel_mean_Features2

        GCN_A1 = AtoGCN_A(WeightMat1)
        GCN_A1 = torch.tensor(GCN_A1, dtype=torch.float32).to(device)
        GCN_A2 = AtoGCN_A(WeightMat2)
        GCN_A2 = torch.tensor(GCN_A2, dtype=torch.float32).to(device)

        superpix_img1 = torch.tensor(superpix_img1, dtype=torch.long).to(device)
        superpix_img2 = torch.tensor(superpix_img2, dtype=torch.long).to(device)

        normLap1 = A2normLap(WeightMat1)
        normLap2 = A2normLap(WeightMat2)
        del WeightMat1, WeightMat2
        normLap1 = torch.tensor(normLap1, dtype=torch.float32).to(device)
        normLap2 = torch.tensor(normLap2, dtype=torch.float32).to(device)

        # net_input = np.array(HyperCube, np.float32)
        net_input = np.array(data, np.float32)
        net_input = torch.from_numpy(net_input.astype(np.float32)).to(device)

        net = DFSGCN_model.DFSGCN_bigdata(data_name, height, width, bands, class_count, GCN_A1, GCN_A2, superpix_img1, superpix_img2, sp_num1, sp_num2)

        print("parameters", net.parameters(), len(list(net.parameters())))
        net.to(device)

        from thop import profile
        flops, params = profile(net, inputs=(net_input,))

        zeros = torch.zeros([m * n]).to(device).float()


        def evaluate_performance(network_output, train_samples_gt, train_samples_gt_onehot, require_AA_KPP=False,
                                 printFlag=True):
            if False == require_AA_KPP:
                with torch.no_grad():
                    available_label_idx = (train_samples_gt != 0).float()  # 有效标签的坐标,用于排除背景
                    available_label_count = available_label_idx.sum()  # 有效标签的个数
                    correct_prediction = torch.where(
                        torch.argmax(network_output, 1) == torch.argmax(train_samples_gt_onehot, 1),
                        available_label_idx, zeros).sum()
                    OA = correct_prediction.cpu() / available_label_count

                    # f1score
                    f1_mean, f1_per_class = calculate_f1_score_simple(network_output, train_samples_gt_onehot,
                                                                      available_label_idx, class_count, epsilon=1e-7)
                    f1_mean = f1_mean.cpu().numpy()
                    f1_per_class = f1_per_class.cpu().numpy()

                    return OA, f1_mean
            else:
                with torch.no_grad():
                    # 计算OA
                    available_label_idx = (train_samples_gt != 0).float()  # 有效标签的坐标,用于排除背景
                    available_label_count = available_label_idx.sum()  # 有效标签的个数
                    correct_prediction = torch.where(
                        torch.argmax(network_output, 1) == torch.argmax(train_samples_gt_onehot, 1),
                        available_label_idx, zeros).sum()
                    OA = correct_prediction.cpu() / available_label_count
                    OA = OA.cpu().numpy()

                    # f1score
                    f1_mean, f1_per_class = calculate_f1_score_simple(network_output, train_samples_gt_onehot,
                                                                      available_label_idx, class_count, epsilon=1e-7)
                    f1_mean = f1_mean.cpu().numpy()
                    f1_per_class = f1_per_class.cpu().numpy()

                    # 计算AA
                    zero_vector = np.zeros([class_count])
                    output_data = network_output.cpu().numpy()
                    train_samples_gt = train_samples_gt.cpu().numpy()
                    train_samples_gt_onehot = train_samples_gt_onehot.cpu().numpy()

                    output_data = np.reshape(output_data, [m * n, class_count])
                    idx = np.argmax(output_data, axis=-1)
                    for z in range(output_data.shape[0]):
                        if ~(zero_vector == output_data[z]).all():
                            idx[z] += 1
                    # idx = idx + train_samples_gt
                    count_perclass = np.zeros([class_count])
                    correct_perclass = np.zeros([class_count])
                    for x in range(len(train_samples_gt)):
                        if train_samples_gt[x] != 0:
                            count_perclass[int(train_samples_gt[x] - 1)] += 1
                            if train_samples_gt[x] == idx[x]:
                                correct_perclass[int(train_samples_gt[x] - 1)] += 1
                    test_AC_list = correct_perclass / count_perclass
                    test_AA = np.average(test_AC_list)

                    # 计算KPP
                    test_pre_label_list = []
                    test_real_label_list = []
                    output_data = np.reshape(output_data, [m * n, class_count])
                    idx = np.argmax(output_data, axis=-1)
                    idx = np.reshape(idx, [m, n])
                    for ii in range(m):
                        for jj in range(n):
                            if Test_GT[ii][jj] != 0:
                                test_pre_label_list.append(idx[ii][jj] + 1)
                                test_real_label_list.append(Test_GT[ii][jj])
                    test_pre_label_list = np.array(test_pre_label_list)
                    test_real_label_list = np.array(test_real_label_list)
                    kappa = metrics.cohen_kappa_score(test_pre_label_list.astype(np.int16),
                                                      test_real_label_list.astype(np.int16))
                    test_kpp = kappa

                    # 输出
                    if printFlag:
                        print("test OA=", OA, "AA=", test_AA, 'kpp=', test_kpp, 'f1_score=', f1_mean)
                        print('acc per class:')
                        print(test_AC_list)
                        print('f1score per class:')
                        print(f1_per_class)

                    OA_ALL.append(OA)
                    F1score_ALL.append(f1_mean)
                    AA_ALL.append(test_AA)
                    KPP_ALL.append(test_kpp)
                    AVG_ALL.append(test_AC_list)

                    if not os.path.exists('./results'):
                        os.makedirs('./results')
                    # 保存数据信息
                    f = open('results\\' + dataset_name + '_results.txt', 'a+')
                    str_results = '\n======================' \
                                  + " learning rate=" + str(learning_rate) \
                                  + " epochs=" + str(max_epoch) \
                                  + " train ratio=" + str(train_ratio) \
                                  + " val ratio=" + str(val_ratio) \
                                  + " ======================" \
                                  + "\nOA=" + str(OA) \
                                  + "\nF1_score=" + str(f1_mean) \
                                  + "\nAA=" + str(test_AA) \
                                  + '\nkpp=' + str(test_kpp) \
                                  + '\nacc per class:' + str(test_AC_list) + "\n" \
                                  + '\nf1score per class:' + str(f1_per_class) + "\n"
                    f.write(str_results)
                    f.close()
                    return OA, test_AA, test_kpp, test_AC_list, f1_mean, f1_per_class


        # 训练
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # ,weight_decay=0.0001
        # optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,weight_decay=0.0001)  # ,weight_decay=0.0001
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        best_loss = 9999999
        best_val_OA=0
        net.train()
        tic1 = time.perf_counter()
        for i in range(max_epoch + 1):
            optimizer.zero_grad()  # zero the gradient buffers
            output, _, _, _ = net(net_input)
            # loss = compute_loss(output, train_samples_gt_onehot, train_label_mask)
            loss = compute_lossV2(output, train_samples_gt_onehot, train_label_mask,
                                superpix_img1,superpix_img2,normLap1,normLap2,sp_num1,sp_num2,i)
            loss.backward(retain_graph=False)
            optimizer.step()  # Does the update
            # scheduler.step()
            if i % 5 == 0:
                with torch.no_grad():
                    net.eval()
                    output, _, _, _ = net(net_input)

                    trainloss = compute_loss(output, train_samples_gt_onehot, train_label_mask)
                    trainOA, train_f1 = evaluate_performance(output, train_samples_gt, train_samples_gt_onehot)
                    valloss = compute_loss(output, val_samples_gt_onehot, val_label_mask)
                    valOA, val_f1 = evaluate_performance(output, val_samples_gt, val_samples_gt_onehot)

                    print(
                        "{}\ttrain loss={}\t train OA={} val loss={}\t val OA={}\t val f1score={}".format(str(i + 1), trainloss, trainOA,
                                                                                         valloss, valOA, val_f1))

                    if valOA > best_val_OA :
                        best_val_OA = valOA
                        torch.save(net.state_dict(), "model\\best_model.pt")
                        print('save model...')
                torch.cuda.empty_cache()
                net.train()

        toc1 = time.perf_counter()
        print("\n\n====================training done. starting evaluation...========================\n")
        training_time = toc1 - tic1  # 分割耗时需要算进去
        Train_Time_ALL.append(training_time)

        torch.cuda.empty_cache()
        with torch.no_grad():
            net.load_state_dict(torch.load("model\\best_model.pt"))
            net.eval()
            tic2 = time.perf_counter()
            output, fused_fea, fea_scale1, fea_scale2 = net(net_input)

            toc2 = time.perf_counter()
            testloss = compute_loss(output, test_samples_gt_onehot, test_label_mask)
            testOA, test_AA, test_kpp, test_AC_list, f1_mean, f1_per_class = evaluate_performance(output, test_samples_gt,
                                                                           test_samples_gt_onehot, require_AA_KPP=True,
                                                                           printFlag=False)
            print("{}\ttest loss={}\t test OA={}\t test AA={}\t test f1score={}".format(str(i + 1), testloss, testOA, test_AA, f1_mean))
            # 计算
            classification_map = torch.argmax(output, 1).reshape([height, width]).cpu() + 1
            Draw_Classification_Map(classification_map, "results\\" + dataset_name + str(testOA))
            testing_time = toc2 - tic2  # 分割耗时需要算进去
            Test_Time_ALL.append(testing_time)

            save_root = './Results/{}/num{}LeakyReLU_251025/iter{}'.format(dataset_name, curr_train_ratio, curr_seed)
            if not os.path.exists(save_root):
                os.makedirs(save_root)

            sio.savemat(save_root + '/training_time.mat', {'training_time': training_time})
            sio.savemat(save_root + '/testing_time.mat', {'testing_time': testing_time})

            #### post-processing
            # median filter
            network_output = output.cpu().numpy().copy()
            network_output = np.reshape(network_output, [m, n, class_count])
            # 对每个通道应用中值滤波
            # scipy 的 medfilt 需要二维数组，所以需要处理每个通道的二维图像
            for iii in range(network_output.shape[2]):  # 遍历通道
                out_test = network_output[:, :, iii]
                network_output[:, :, iii] = medfilt(out_test, kernel_size=kernel_size)
            # 将结果从 NumPy 转回 Tensor，并移回 GPU
            network_output = np.reshape(network_output, [m * n, class_count])
            output = torch.tensor(network_output, dtype=torch.float32).to(device)

            testloss = compute_loss(output, test_samples_gt_onehot, test_label_mask)
            testOA, test_AA, test_kpp, test_AC_list, f1_mean, f1_per_class = evaluate_performance(output, test_samples_gt,
                                                                           test_samples_gt_onehot, require_AA_KPP=True,
                                                                           printFlag=False)
            # print("{}\ttest loss={}\t test OA={}\t test AA={}".format(str(i + 1), testloss, testOA, test_AA))
            print("{}\ttest loss={}\t test OA={}\t test AA={}\t test f1score={}".format(str(i + 1), testloss, testOA, test_AA, f1_mean))
            # 计算
            classification_map = torch.argmax(output, 1).reshape([height, width]).cpu() + 1
            Draw_Classification_Map(classification_map, "results\\" + dataset_name + str(testOA))

            # Saving data
            sio.savemat(save_root + '/PredLabels_med.mat', {'classification_map': classification_map.numpy()})
            all_output = np.append(test_AC_list, [testOA, test_AA, test_kpp, f1_mean])
            sio.savemat(save_root + '/all_output_med.mat', {'all_output': all_output})
            sio.savemat(save_root + '/f1_mean_med.mat', {'f1_mean': f1_mean})

            OA_med_ALL.append(testOA)
            AA_med_ALL.append(test_AA)
            kappa_med_ALL.append(test_kpp)
            F1score_med_ALL.append(f1_mean)
            AVG_med_ALL.append(test_AC_list)
            F1perclass_med_ALL.append(f1_per_class)

        torch.cuda.empty_cache()
        del net

    OA_med_ALL = np.array(OA_med_ALL)
    AA_med_ALL = np.array(AA_med_ALL)
    kappa_med_ALL = np.array(kappa_med_ALL)
    F1score_med_ALL = np.array(F1score_med_ALL)
    AVG_med_ALL = np.array(AVG_med_ALL)
    F1perclass_med_ALL = np.array(F1perclass_med_ALL)
    Train_Time_ALL = np.array(Train_Time_ALL)
    Test_Time_ALL = np.array(Test_Time_ALL)

    print("\ntrain_ratio={}".format(curr_train_ratio),
          "\n==============================================================================")
    print('OA_med=', np.mean(OA_med_ALL), '+-', np.std(OA_med_ALL))
    print('F1score=', np.mean(F1score_med_ALL), '+-', np.std(F1score_med_ALL))
    print('AA=', np.mean(AA_med_ALL), '+-', np.std(AA_med_ALL))
    print('Kpp=', np.mean(kappa_med_ALL), '+-', np.std(kappa_med_ALL))
    print('acc_each_class=', np.mean(AVG_med_ALL, 0), '+-', np.std(AVG_med_ALL, 0))
    print('f1score_each_class=', np.mean(F1perclass_med_ALL, 0), '+-', np.std(F1perclass_med_ALL, 0))
    print("Average training time:{}".format(np.mean(Train_Time_ALL)))
    print("Average testing time:{}".format(np.mean(Test_Time_ALL)))

    # 保存数据信息
    f = open('results\\' + dataset_name + '_results.txt', 'a+')
    str_results = '\n\n************************************************' \
                  + "\ntrain_ratio={}".format(curr_train_ratio) \
                  + '\nOA=' + str(np.mean(OA_med_ALL)) + '+-' + str(np.std(OA_med_ALL)) \
                  + '\nF1score=' + str(np.mean(F1score_med_ALL)) + '+-' + str(np.std(F1score_med_ALL)) \
                  + '\nAA=' + str(np.mean(AA_med_ALL)) + '+-' + str(np.std(AA_med_ALL)) \
                  + '\nKpp=' + str(np.mean(kappa_med_ALL)) + '+-' + str(np.std(kappa_med_ALL)) \
                  + '\nacc_each_class=' + str(np.mean(AVG_med_ALL, 0)) + '+-' + str(np.std(AVG_med_ALL, 0)) \
                  + '\nf1score_each_class=' + str(np.mean(F1perclass_med_ALL, 0)) + '+-' + str(np.std(F1perclass_med_ALL, 0)) \
                  + "\nAverage training time:{}".format(np.mean(Train_Time_ALL)) \
                  + "\nAverage testing time:{}".format(np.mean(Test_Time_ALL))
    f.write(str_results)
    f.close()






