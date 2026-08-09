import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GCNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(GCNLayer, self).__init__()
        self.BN = nn.BatchNorm1d(input_dim)
        self.Activition = nn.LeakyReLU()
        self.sigma1= torch.nn.Parameter(torch.tensor([0.1],requires_grad=True))
        # 第一层GCN
        self.GCN_liner_theta_1 =nn.Sequential(nn.Linear(input_dim, 256))
        self.GCN_liner_out_1 =nn.Sequential( nn.Linear(input_dim, output_dim))
        self.GCN_liner_out_2 =nn.Sequential( nn.Linear(input_dim, output_dim))
        
        
    def A_to_D_inv(self, A: torch.Tensor):
        D = A.sum(1)
        D_hat = torch.diag(torch.pow(D, -0.5))
        return D_hat
    
    def forward(self, H, A, model='normal'):
        # # 方案一：minmax归一化
        # H = self.BN(H)
        # H_xx1= self.GCN_liner_theta_1(H)
        # A = torch.clamp(torch.sigmoid(torch.matmul(H_xx1, H_xx1.t())), min=0.1) * self.mask + self.I
        # if model != 'normal': A=torch.clamp(A,0.1) #This is a trick.
        # D_hat = self.A_to_D_inv(A)
        # A_hat = torch.matmul(D_hat, torch.matmul(A,D_hat))
        # output = torch.mm(A_hat, self.GCN_liner_out_1(H))
        # output = self.Activition(output)
        
        # # 方案二：softmax归一化 (加速运算)
        H = self.BN(H)
        # H_xx1= self.GCN_liner_theta_1(H)
        # e = torch.sigmoid(torch.matmul(H_xx1, H_xx1.t()))
        # zero_vec = -9e15 * torch.ones_like(e)
        # nodes_count=A.shape[0]
        # I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)
        # A = torch.where(A > 0, e, zero_vec)+ I
        # del I, e, zero_vec
        # if model != 'normal': A=torch.clamp(A,0.1) #This is a trick for the Indian Pines.
        # A = A + I
        # A = F.softmax(A, dim=1)
        output = self.Activition(torch.mm(A, self.GCN_liner_out_1(H)))
        output = self.Activition(torch.mm(A, self.GCN_liner_out_1(H))) + self.Activition(self.GCN_liner_out_2(H))

        return output,A


class GCNLayerV2(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(GCNLayerV2, self).__init__()
        self.BN = nn.BatchNorm1d(input_dim)
        self.Activition = nn.LeakyReLU()
        # self.Activition = nn.ReLU()
        self.sigma1 = torch.nn.Parameter(torch.tensor([0.1], requires_grad=True))
        # 第一层GCN
        self.GCN_liner_theta_1 = nn.Sequential(nn.Linear(input_dim, 256))
        self.GCN_liner_out_1 = nn.Sequential(nn.Linear(input_dim, output_dim))
        self.GCN_liner_out_2 = nn.Sequential(nn.Linear(input_dim, output_dim))

    def A_to_D_inv(self, A: torch.Tensor):
        D = A.sum(1)
        D_hat = torch.diag(torch.pow(D, -0.5))
        return D_hat

    def forward(self, H, A, model='normal'):
        # 方案一：minmax归一化
        # H = self.BN(H)
        # # H_xx1= self.GCN_liner_theta_1(H)               # comment or not
        # H_xx1= H          # comment or not
        # nodes_count = A.shape[0]
        # # A = torch.clamp(torch.sigmoid(torch.matmul(H_xx1, H_xx1.t())), min=0.1) * self.mask + self.I
        # e = torch.sigmoid(torch.matmul(H_xx1, H_xx1.t()))
        # e = torch.clamp(e, min=0.1)    # comment or not
        # zero_vec = 0 * torch.ones_like(e)
        # I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)
        # A = torch.where(A > 0, e, zero_vec) + I
        # # if model != 'normal': A=torch.clamp(A,0.1) #This is a trick.
        # D_hat = self.A_to_D_inv(A)
        # A_hat = torch.matmul(D_hat, torch.matmul(A,D_hat))
        # output = torch.mm(A_hat, self.GCN_liner_out_1(H))
        # output = self.Activition(output)

        # # 方案二：softmax归一化 (加速运算)
        H = self.BN(H)
        # H_xx1 = self.GCN_liner_theta_1(H)
        # e = torchtmul(H_.sigmoid(torch.maxx1, H_xx1.t()))
        H_xx1= H
        # e = torch.matmul(H_xx1, H_xx1.t())
        e = torch.sigmoid(torch.matmul(H_xx1, H_xx1.t()))
        zero_vec = -9e15 * torch.ones_like(e)
        nodes_count = A.shape[0]
        I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)
        A = torch.where(A > 0, e, zero_vec) + I
        del I, e, zero_vec
        # if model != 'normal': A=torch.clamp(A,0.1) #This is a trick for the Indian Pines.
        A = F.softmax(A, dim=1)
        output = self.Activition(torch.mm(A, self.GCN_liner_out_1(H))) + self.Activition(self.GCN_liner_out_2(H))
        # output = self.Activition(torch.mm(A, self.GCN_liner_out_1(H)))

        return output, A

class GCNLayerV3(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(GCNLayerV3, self).__init__()
        self.BN = nn.BatchNorm1d(input_dim)
        self.Activition = nn.LeakyReLU()
        # self.Activition = nn.ReLU()
        self.sigma1 = torch.nn.Parameter(torch.tensor([0.1], requires_grad=True))
        # 第一层GCN
        self.GCN_liner_theta_1 = nn.Sequential(nn.Linear(input_dim, 256))
        self.GCN_liner_out_1 = nn.Sequential(nn.Linear(input_dim, output_dim))
        self.GCN_liner_out_2 = nn.Sequential(nn.Linear(input_dim, output_dim))
        self.BN1 = nn.BatchNorm1d(output_dim)
        self.BN2 = nn.BatchNorm1d(output_dim)

    def A_to_D_inv(self, A: torch.Tensor):
        D = A.sum(1)
        D_hat = torch.diag(torch.pow(D, -0.5))
        return D_hat

    def forward(self, H, A, model='normal'):
        # # 方案一：minmax归一化
        # H = self.BN(H)
        # H_xx1= self.GCN_liner_theta_1(H)
        # A = torch.clamp(torch.sigmoid(torch.matmul(H_xx1, H_xx1.t())), min=0.1) * self.mask + self.I
        # if model != 'normal': A=torch.clamp(A,0.1) #This is a trick.
        # D_hat = self.A_to_D_inv(A)
        # A_hat = torch.matmul(D_hat, torch.matmul(A,D_hat))
        # output = torch.mm(A_hat, self.GCN_liner_out_1(H))
        # output = self.Activition(output)

        # # 方案二：softmax归一化 (加速运算)
        H = self.BN(H)
        # H_xx1 = self.GCN_liner_theta_1(H)
        # e = torch.sigmoid(torch.matmul(H_xx1, H_xx1.t()))
        H_xx1= H
        # e = torch.matmul(H_xx1, H_xx1.t())
        e = torch.sigmoid(torch.matmul(H_xx1, H_xx1.t()))
        zero_vec = -9e15 * torch.ones_like(e)
        nodes_count = A.shape[0]
        I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)
        A = torch.where(A > 0, e, zero_vec) + I
        del I, e, zero_vec
        # if model != 'normal': A=torch.clamp(A,0.1) #This is a trick for the Indian Pines.
        A = F.softmax(A, dim=1)
        output1 = torch.mm(A, self.GCN_liner_out_1(H))
        output1 = self.BN1(output1)
        output2 = self.GCN_liner_out_2(H)
        output2 = self.BN2(output2)
        # output = self.Activition(torch.mm(A, self.GCN_liner_out_1(H)))

        output = self.Activition(output1+output2)
        return output, A

class GCNLayerV4(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(GCNLayerV4, self).__init__()
        self.BN = nn.BatchNorm1d(input_dim)
        self.Activition = nn.LeakyReLU()
        # self.Activition = nn.ReLU()
        self.sigma1 = torch.nn.Parameter(torch.tensor([0.1], requires_grad=True))
        # 第一层GCN
        self.GCN_liner_theta_1 = nn.Sequential(nn.Linear(input_dim, 256))
        self.GCN_liner_out_1 = nn.Sequential(nn.Linear(input_dim, output_dim))
        self.BN1 = nn.BatchNorm1d(output_dim)

    def A_to_D_inv(self, A: torch.Tensor):
        D = A.sum(1)
        D_hat = torch.diag(torch.pow(D, -0.5))
        return D_hat

    def forward(self, H, A, model='normal'):
        # # 方案一：minmax归一化
        # H = self.BN(H)
        # H_xx1= self.GCN_liner_theta_1(H)
        # A = torch.clamp(torch.sigmoid(torch.matmul(H_xx1, H_xx1.t())), min=0.1) * self.mask + self.I
        # if model != 'normal': A=torch.clamp(A,0.1) #This is a trick.
        # D_hat = self.A_to_D_inv(A)
        # A_hat = torch.matmul(D_hat, torch.matmul(A,D_hat))
        # output = torch.mm(A_hat, self.GCN_liner_out_1(H))
        # output = self.Activition(output)

        # # 方案二：softmax归一化 (加速运算)
        H = self.BN(H)
        # H_xx1 = self.GCN_liner_theta_1(H)
        # e = torch.sigmoid(torch.matmul(H_xx1, H_xx1.t()))
        H_xx1= H
        # e = torch.matmul(H_xx1, H_xx1.t())
        e = torch.sigmoid(torch.matmul(H_xx1, H_xx1.t()))
        zero_vec = -9e15 * torch.ones_like(e)
        nodes_count = A.shape[0]
        I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)
        A = torch.where(A > 0, e, zero_vec) + I
        del I, e, zero_vec
        # if model != 'normal': A=torch.clamp(A,0.1) #This is a trick for the Indian Pines.
        A = F.softmax(A, dim=1)
        output1 = torch.mm(A, self.GCN_liner_out_1(H))
        output1 = self.BN1(output1)
        # output = self.Activition(torch.mm(A, self.GCN_liner_out_1(H)))

        output = self.Activition(output1)
        return output, A


class SSConv(nn.Module):
    '''
    Spectral-Spatial Convolution
    '''
    def __init__(self, in_ch, out_ch,kernel_size=3):
        super(SSConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            groups=out_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.Act1 = nn.LeakyReLU()
        self.Act2 = nn.LeakyReLU()
        self.BN=nn.BatchNorm2d(in_ch)
        
    
    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.Act1(out)
        out = self.depth_conv(out)
        out = self.Act2(out)
        return out


class DFSGCN(nn.Module):
    def __init__(self, height: int, width: int, channels: int, class_count: int, A1: torch.Tensor,
                 A2: torch.Tensor, superpix_img1: torch.Tensor, superpix_img2: torch.Tensor, Q1: torch.Tensor, Q2: torch.Tensor):
        super(DFSGCN, self).__init__()
        # 类别数,即网络最终输出通道数
        self.class_count = class_count  # 类别数
        # 网络输入数据大小
        self.channel = channels
        self.Height = height
        self.Width = width
        self.A1 = A1
        self.A2 = A2
        self.Q1 = Q1
        self.Q2 = Q2
        self.superpix_img1 = superpix_img1
        self.superpix_img2 = superpix_img2

        N1 = 2  # 输出卷积核数
        input_channel = self.channel
        # input_channel = N1*self.channel
        layers_count = 1
        # Superpixel-level Graph Sub-Network
        self.GCN_Branch1 = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.GCN_Branch1.add_module('GCN_Branch' + str(i), GCNLayerV2(input_channel, 128))
            elif (i>0) and (i < layers_count - 1):
                self.GCN_Branch1.add_module('GCN_Branch' + str(i), GCNLayer(128, 128))
            else:
                # self.GCN_Branch1.add_module('GCN_Branch' + str(i), GCNLayer(128, self.class_count))
                self.GCN_Branch1.add_module('GCN_Branch' + str(i), GCNLayerV2(128, 64))
                # self.GCN_Branch.add_module('GCN_Branch' + str(i), GCNLayer(128, 128))

        self.GCN_Branch2 = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.GCN_Branch2.add_module('GCN_Branch' + str(i), GCNLayerV2(input_channel, 128))
            elif (i>0) and (i < layers_count - 1):
                self.GCN_Branch2.add_module('GCN_Branch' + str(i), GCNLayer(128, 128))
            else:
                # self.GCN_Branch2.add_module('GCN_Branch' + str(i), GCNLayer(128, self.class_count))
                self.GCN_Branch2.add_module('GCN_Branch' + str(i), GCNLayerV2(128, 64))
                # self.GCN_Branch.add_module('GCN_Branch' + str(i), GCNLayer(128, 128))

        self.BN1 = nn.BatchNorm1d(128)
        self.GCN_Branch3 = GCNLayer(128, self.class_count)
        self.BN2 = nn.BatchNorm1d(128)
        # Softmax layer
        self.Softmax_linear = nn.Sequential(nn.Linear(128, self.class_count))
        self.Softmax_linear1 = nn.Sequential(nn.Linear(64, self.class_count))
        self.Softmax_linear2 = nn.Sequential(nn.Linear(64, self.class_count))
        self.Softmax_linear3 = nn.Sequential(nn.Linear(3*self.class_count, self.class_count))
        # self.Softmax_linear = nn.Sequential(nn.Linear(64, self.class_count))

        self.BN_H1 = nn.BatchNorm1d(self.channel)
        self.BN_H2 = nn.BatchNorm1d(self.channel)
        self.conv1d_H1 = nn.Conv1d(in_channels=1, out_channels=N1, kernel_size=10, stride=1,padding='same')
        self.conv1d_H2 = nn.Conv1d(in_channels=1, out_channels=N1, kernel_size=10, stride=1,padding='same')

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        '''
        :param x: H*W*C
        :return: probability_map
        '''
        # GCN层 1 转化为超像素 x_flat 乘以 列归一化Q
        # H = superpixels_flatten

        # H1 = self.BN_H1(x1)
        # H1_bn = H1.unsqueeze(1)  # 变形为 (N, input_channels, 1)
        # # 计算卷积
        # H1_bn_zj = self.conv1d_H1(H1_bn)  # 输出形状 (N, N1, 1)
        # # 将输出展平为 (N, N1 * input_channels) 形状
        # H1 = H1_bn_zj.view(x1.shape[0], -1)  # 展平
        #
        # H2 = self.BN_H2(x2)
        # H2_bn = H2.unsqueeze(1)  # 变形为 (N, input_channels, 1)
        # # 计算卷积
        # H2_bn_zj = self.conv1d_H2(H2_bn)  # 输出形状 (N, N1, 1)
        # # 将输出展平为 (N, N1 * input_channels) 形状
        # H2 = H2_bn_zj.view(x2.shape[0], -1)  # 展平

        H1 = x1
        for i in range(len(self.GCN_Branch1)):
            H1, _ = self.GCN_Branch1[i](H1, self.A1)
        H2 = x2
        for i in range(len(self.GCN_Branch2)):
            H2, _ = self.GCN_Branch2[i](H2, self.A2)


        # # 初始化Posterior_PPSPG_ALL2矩阵
        # Posterior_PPSPG_ALL1 = torch.zeros(self.Width * self.Height, self.class_count)
        #
        # # 遍历每个超像素块
        # for select_i in range(1, H1.shape[0] + 1):  # Python的索引从0开始，MATLAB从1开始，因此这里从1到sp_num2
        #     # 获取对应超像素块的位置索引
        #     pos_idx = (self.superpix_img1 == select_i).nonzero()[0]  # 获取所有匹配的坐标索引
        #
        #     # 将Posterior_PPSPGv2中的行重复填充到Posterior_PPSPG_ALL2的相应位置
        #     Posterior_PPSPG_ALL1[pos_idx, :] = H1[select_i - 1, :].repeat(len(pos_idx), 1)
        #
        # # 初始化Posterior_PPSPG_ALL2矩阵
        # Posterior_PPSPG_ALL2 = torch.zeros(self.Width * self.Height, self.class_count)
        #
        # # 遍历每个超像素块
        # for select_i in range(1, H2.shape[0] + 1):  # Python的索引从0开始，MATLAB从1开始，因此这里从1到sp_num2
        #     # 获取对应超像素块的位置索引
        #     pos_idx = (self.superpix_img2 == select_i).nonzero(as_tuple=True)[0]  # 获取所有匹配的坐标索引
        #
        #     # 将Posterior_PPSPGv2中的行重复填充到Posterior_PPSPG_ALL2的相应位置
        #     Posterior_PPSPG_ALL2[pos_idx, :] = H2[select_i - 1, :].repeat(len(pos_idx), 1)

        # H1 = self.Softmax_linear(H1)
        # H2 = self.Softmax_linear(H2)
        # 使用矩阵乘法来计算 Posterior_PPSPG_ALL1
        Posterior_PPSPG_ALL1 = torch.matmul(self.Q1, H1)
        Posterior_PPSPG_ALL2 = torch.matmul(self.Q2, H2)
        # GCN_result = torch.matmul(self.Q, H)  # 这里self.norm_row_Q == self.Q

        # GCN_result = Posterior_PPSPG_ALL1*Posterior_PPSPG_ALL2  # 这里self.norm_row_Q == self.Q
        GCN_result = torch.cat([Posterior_PPSPG_ALL1, Posterior_PPSPG_ALL2], dim=-1)
        # GCN_result,_ = self.GCN_Branch2(H, self.A)

        # 两组特征融合(两种融合方式)
        # Y = torch.cat([GCN_result, CNN_result], dim=-1)
        Y0 = GCN_result
        Y0 = self.Softmax_linear(Y0)
        # Y1 = self.Softmax_linear1(Posterior_PPSPG_ALL1)
        # Y2 = self.Softmax_linear2(Posterior_PPSPG_ALL2)
        # Y = torch.cat([Y0, Y1, Y2], dim=-1)
        # Y = self.Softmax_linear3(Y)
        Y = Y0
        Y = F.softmax(Y, -1)
        return Y


class DFSGCN_V2(nn.Module):
    def __init__(self, data_name: str, height: int, width: int, channels: int, class_count: int, A1: torch.Tensor,
                 A2: torch.Tensor, superpix_img1: torch.Tensor, superpix_img2: torch.Tensor, Q1: torch.Tensor, Q2: torch.Tensor):
        super(DFSGCN_V2, self).__init__()
        # 类别数,即网络最终输出通道数
        self.class_count = class_count  # 类别数
        # 网络输入数据大小
        self.channel = channels
        self.Height = height
        self.Width = width
        self.A1 = A1
        self.A2 = A2
        self.Q1 = Q1
        self.Q2 = Q2
        self.superpix_img1 = superpix_img1
        self.superpix_img2 = superpix_img2
        self.norm_col_Q1 = Q1 / (torch.sum(Q1, 0, keepdim=True))  # 列归一化Q
        self.norm_col_Q2 = Q2 / (torch.sum(Q2, 0, keepdim=True))  # 列归一化Q

        if data_name == "Zaoyuan2":
            layers_count = 2   # Zaoyuan
        else:
            layers_count = 1   # other data

        # Spectra Transformation Sub-Network
        self.CNN_denoise = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(self.channel))
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i),
                                            nn.Conv2d(self.channel, 128, kernel_size=(1, 1)))
                # self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i),
                #                             nn.Conv2d(self.channel, 128, kernel_size=(3, 3),padding=(1,1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
            else:
                self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(128), )
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(128, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())

        layers_count = 2
        # Spectra Transformation Sub-Network
        self.CNN_denoise2 = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.CNN_denoise2.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(self.channel))
                self.CNN_denoise2.add_module('CNN_denoise_Conv' + str(i),
                                            nn.Conv2d(self.channel, 128, kernel_size=(1, 1)))
                self.CNN_denoise2.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
            else:
                self.CNN_denoise2.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(128), )
                self.CNN_denoise2.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(128, 128, kernel_size=(1, 1)))
                self.CNN_denoise2.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())

        # postprocessing network
        self.CNN_denoise3 = nn.Sequential()
        self.CNN_denoise3.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(self.class_count))
        self.CNN_denoise3.add_module('CNN_denoise_Conv' + str(i),
                                    nn.Conv2d(self.class_count, self.class_count, kernel_size=(3, 3),padding=(1, 1)))
        self.CNN_denoise3.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())

        N1 = 2  # 输出卷积核数
        # input_channel = self.channel
        # input_channel = N1*self.channel
        input_channel = 128
        layers_count = 2
        # Superpixel-level Graph Sub-Network
        self.GCN_Branch1 = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.GCN_Branch1.add_module('GCN_Branch' + str(i), GCNLayerV2(input_channel, 128))
            elif (i>0) and (i < layers_count - 1):
                self.GCN_Branch1.add_module('GCN_Branch' + str(i), GCNLayer(128, 128))
            else:
                # self.GCN_Branch1.add_module('GCN_Branch' + str(i), GCNLayer(128, self.class_count))
                self.GCN_Branch1.add_module('GCN_Branch' + str(i), GCNLayerV2(128, 64))
                # self.GCN_Branch.add_module('GCN_Branch' + str(i), GCNLayer(128, 128))

        self.GCN_Branch2 = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.GCN_Branch2.add_module('GCN_Branch' + str(i), GCNLayerV2(input_channel, 128))
            elif (i>0) and (i < layers_count - 1):
                self.GCN_Branch2.add_module('GCN_Branch' + str(i), GCNLayer(128, 128))
            else:
                # self.GCN_Branch2.add_module('GCN_Branch' + str(i), GCNLayer(128, self.class_count))
                self.GCN_Branch2.add_module('GCN_Branch' + str(i), GCNLayerV2(128, 64))
                # self.GCN_Branch.add_module('GCN_Branch' + str(i), GCNLayer(128, 128))

        self.BN1 = nn.BatchNorm1d(128)
        self.GCN_Branch3 = GCNLayer(128, self.class_count)
        self.BN2 = nn.BatchNorm1d(128)
        # Softmax layer
        self.Softmax_linear = nn.Sequential(nn.Linear(128, self.class_count))
        self.Softmax_linear1 = nn.Sequential(nn.Linear(64, self.class_count))
        self.Softmax_linear2 = nn.Sequential(nn.Linear(64, self.class_count))
        self.Softmax_linear3 = nn.Sequential(nn.Linear(3*self.class_count, self.class_count))
        # self.Softmax_linear = nn.Sequential(nn.Linear(64, self.class_count))

        self.BN_H1 = nn.BatchNorm1d(self.channel)
        self.BN_H2 = nn.BatchNorm1d(self.channel)
        self.conv1d_H1 = nn.Conv1d(in_channels=1, out_channels=N1, kernel_size=10, stride=1,padding='same')
        self.conv1d_H2 = nn.Conv1d(in_channels=1, out_channels=N1, kernel_size=10, stride=1,padding='same')

    def forward(self, x: torch.Tensor):
        '''
        :param x: H*W*C
        :return: probability_map
        '''
        (h, w, c) = x.shape

        # 先去除噪声
        noise = self.CNN_denoise(torch.unsqueeze(x.permute([2, 0, 1]), 0))
        noise = torch.squeeze(noise, 0).permute([1, 2, 0])
        clean_x = noise  # 直连

        clean_x_flatten = clean_x.reshape([h * w, -1])

        # noise2 = self.CNN_denoise2(torch.unsqueeze(x.permute([2, 0, 1]), 0))
        # noise2 = torch.squeeze(noise2, 0).permute([1, 2, 0])
        # clean_x2 = noise2  # 直连
        #
        # clean_x_flatten2 = clean_x2.reshape([h * w, -1])

        x1 = torch.mm(self.norm_col_Q1.t(), clean_x_flatten)  # 低频部分
        x2 = torch.mm(self.norm_col_Q2.t(), clean_x_flatten)  # 低频部分


        H1 = x1
        for i in range(len(self.GCN_Branch1)):
            H1, _ = self.GCN_Branch1[i](H1, self.A1)
        H2 = x2
        for i in range(len(self.GCN_Branch2)):
            H2, _ = self.GCN_Branch2[i](H2, self.A2)


        # # 初始化Posterior_PPSPG_ALL2矩阵
        # Posterior_PPSPG_ALL1 = torch.zeros(self.Width * self.Height, self.class_count)
        #
        # # 遍历每个超像素块
        # for select_i in range(1, H1.shape[0] + 1):  # Python的索引从0开始，MATLAB从1开始，因此这里从1到sp_num2
        #     # 获取对应超像素块的位置索引
        #     pos_idx = (self.superpix_img1 == select_i).nonzero()[0]  # 获取所有匹配的坐标索引
        #
        #     # 将Posterior_PPSPGv2中的行重复填充到Posterior_PPSPG_ALL2的相应位置
        #     Posterior_PPSPG_ALL1[pos_idx, :] = H1[select_i - 1, :].repeat(len(pos_idx), 1)
        #
        # # 初始化Posterior_PPSPG_ALL2矩阵
        # Posterior_PPSPG_ALL2 = torch.zeros(self.Width * self.Height, self.class_count)
        #
        # # 遍历每个超像素块
        # for select_i in range(1, H2.shape[0] + 1):  # Python的索引从0开始，MATLAB从1开始，因此这里从1到sp_num2
        #     # 获取对应超像素块的位置索引
        #     pos_idx = (self.superpix_img2 == select_i).nonzero(as_tuple=True)[0]  # 获取所有匹配的坐标索引
        #
        #     # 将Posterior_PPSPGv2中的行重复填充到Posterior_PPSPG_ALL2的相应位置
        #     Posterior_PPSPG_ALL2[pos_idx, :] = H2[select_i - 1, :].repeat(len(pos_idx), 1)

        # H1 = self.Softmax_linear(H1)
        # H2 = self.Softmax_linear(H2)
        # 使用矩阵乘法来计算 Posterior_PPSPG_ALL1
        Posterior_PPSPG_ALL1 = torch.matmul(self.Q1, H1)
        Posterior_PPSPG_ALL2 = torch.matmul(self.Q2, H2)
        # GCN_result = torch.matmul(self.Q, H)  # 这里self.norm_row_Q == self.Q

        # GCN_result = Posterior_PPSPG_ALL1*Posterior_PPSPG_ALL2  # 这里self.norm_row_Q == self.Q
        GCN_result = torch.cat([Posterior_PPSPG_ALL1, Posterior_PPSPG_ALL2], dim=-1)
        # GCN_result,_ = self.GCN_Branch2(H, self.A)

        # 两组特征融合(两种融合方式)
        # Y = torch.cat([GCN_result, CNN_result], dim=-1)
        Y0 = GCN_result
        Y0 = self.Softmax_linear(Y0)
        # Y1 = self.Softmax_linear1(Posterior_PPSPG_ALL1)
        # Y2 = self.Softmax_linear2(Posterior_PPSPG_ALL2)
        # Y = torch.cat([Y0, Y1, Y2], dim=-1)
        # Y = self.Softmax_linear3(Y)

        # post-processing: 2D CNN denoising
        # Y0 = Y0.reshape([h, w, -1])
        # Y0 = self.CNN_denoise3(torch.unsqueeze(Y0.permute([2, 0, 1]), 0))
        # Y0 = torch.squeeze(Y0, 0).permute([1, 2, 0])
        # Y0 = Y0.reshape([h * w, -1])

        Y = Y0
        Y = F.softmax(Y, -1)
        return Y, Y0, Posterior_PPSPG_ALL1, Posterior_PPSPG_ALL2

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

class DFSGCN_bigdata(nn.Module):
    def __init__(self, data_name: str, height: int, width: int, channels: int, class_count: int, A1: torch.Tensor,
                 A2: torch.Tensor, superpix_img1: torch.Tensor, superpix_img2: torch.Tensor, sp_num1: int, sp_num2: int):
        super(DFSGCN_bigdata, self).__init__()
        # 类别数,即网络最终输出通道数
        self.class_count = class_count  # 类别数
        # 网络输入数据大小
        self.channel = channels
        self.Height = height
        self.Width = width
        self.A1 = A1
        self.A2 = A2
        self.superpix_img1 = superpix_img1
        self.superpix_img2 = superpix_img2
        self.sp_num1 = sp_num1
        self.sp_num2 = sp_num2

        if data_name == "Zaoyuan2":
            layers_count = 2   # Zaoyuan
        else:
            layers_count = 1   # other data

        # Spectra Transformation Sub-Network
        self.CNN_denoise = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(self.channel))
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i),
                                            nn.Conv2d(self.channel, 128, kernel_size=(1, 1)))
                # self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i),
                #                             nn.Conv2d(self.channel, 128, kernel_size=(3, 3),padding=(1,1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
            else:
                self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(128), )
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(128, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())

        layers_count = 2
        # Spectra Transformation Sub-Network
        self.CNN_denoise2 = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.CNN_denoise2.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(self.channel))
                self.CNN_denoise2.add_module('CNN_denoise_Conv' + str(i),
                                            nn.Conv2d(self.channel, 128, kernel_size=(1, 1)))
                self.CNN_denoise2.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
            else:
                self.CNN_denoise2.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(128), )
                self.CNN_denoise2.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(128, 128, kernel_size=(1, 1)))
                self.CNN_denoise2.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())

        N1 = 2  # 输出卷积核数
        # input_channel = self.channel
        # input_channel = N1*self.channel
        input_channel = 128
        layers_count = 2
        # Superpixel-level Graph Sub-Network
        self.GCN_Branch1 = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.GCN_Branch1.add_module('GCN_Branch' + str(i), GCNLayerV2(input_channel, 128))
            elif (i>0) and (i < layers_count - 1):
                self.GCN_Branch1.add_module('GCN_Branch' + str(i), GCNLayer(128, 128))
            else:
                # self.GCN_Branch1.add_module('GCN_Branch' + str(i), GCNLayer(128, self.class_count))
                self.GCN_Branch1.add_module('GCN_Branch' + str(i), GCNLayerV2(128, 64))
                # self.GCN_Branch.add_module('GCN_Branch' + str(i), GCNLayer(128, 128))

        self.GCN_Branch2 = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.GCN_Branch2.add_module('GCN_Branch' + str(i), GCNLayerV2(input_channel, 128))
            elif (i>0) and (i < layers_count - 1):
                self.GCN_Branch2.add_module('GCN_Branch' + str(i), GCNLayer(128, 128))
            else:
                # self.GCN_Branch2.add_module('GCN_Branch' + str(i), GCNLayer(128, self.class_count))
                self.GCN_Branch2.add_module('GCN_Branch' + str(i), GCNLayerV2(128, 64))
                # self.GCN_Branch.add_module('GCN_Branch' + str(i), GCNLayer(128, 128))

        self.BN1 = nn.BatchNorm1d(128)
        self.GCN_Branch3 = GCNLayer(128, self.class_count)
        self.BN2 = nn.BatchNorm1d(128)
        # Softmax layer
        self.Softmax_linear = nn.Sequential(nn.Linear(128, self.class_count))

    def forward(self, x: torch.Tensor):
        '''
        :param x: H*W*C
        :return: probability_map
        '''
        (h, w, c) = x.shape

        # 先去除噪声
        noise = self.CNN_denoise(torch.unsqueeze(x.permute([2, 0, 1]), 0))
        noise = torch.squeeze(noise, 0).permute([1, 2, 0])
        clean_x = noise  # 直连

        clean_x_flatten = clean_x.reshape([h * w, -1])

        # x1 = torch.mm(self.norm_col_Q1.t(), clean_x_flatten)  # 低频部分
        # x2 = torch.mm(self.norm_col_Q2.t(), clean_x_flatten)  # 低频部分

        x1 = superpixel_mean(clean_x_flatten, self.superpix_img1, self.sp_num1)
        x2 = superpixel_mean(clean_x_flatten, self.superpix_img2, self.sp_num2)


        H1 = x1
        for i in range(len(self.GCN_Branch1)):
            H1, _ = self.GCN_Branch1[i](H1, self.A1)
        H2 = x2
        for i in range(len(self.GCN_Branch2)):
            H2, _ = self.GCN_Branch2[i](H2, self.A2)

        # 使用矩阵乘法来计算 Posterior_PPSPG_ALL1
        # Posterior_PPSPG_ALL1 = torch.matmul(self.Q1, H1)
        # Posterior_PPSPG_ALL2 = torch.matmul(self.Q2, H2)
        Posterior_PPSPG_ALL1 = H1[self.superpix_img1 - 1,:]  # (H*W) x 32
        Posterior_PPSPG_ALL2 = H2[self.superpix_img2 - 1,:]  # (H*W) x 32
        # GCN_result = torch.matmul(self.Q, H)  # 这里self.norm_row_Q == self.Q

        # GCN_result = Posterior_PPSPG_ALL1*Posterior_PPSPG_ALL2  # 这里self.norm_row_Q == self.Q
        GCN_result = torch.cat([Posterior_PPSPG_ALL1, Posterior_PPSPG_ALL2], dim=-1)

        Y0 = GCN_result
        Y0 = self.Softmax_linear(Y0)

        Y = Y0
        Y = F.softmax(Y, -1)
        return Y, Y0, Posterior_PPSPG_ALL1, Posterior_PPSPG_ALL2

class DFSGCN_V2_lit(nn.Module):
    def __init__(self, height: int, width: int, channels: int, class_count: int, A1: torch.Tensor,
                 A2: torch.Tensor, superpix_img1: torch.Tensor, Q1: torch.Tensor):
        super(DFSGCN_V2_lit, self).__init__()
        # 类别数,即网络最终输出通道数
        self.class_count = class_count  # 类别数
        # 网络输入数据大小
        self.channel = channels
        self.Height = height
        self.Width = width
        self.A1 = A1
        self.A2 = A2
        self.Q1 = Q1
        self.superpix_img1 = superpix_img1
        self.norm_col_Q1 = Q1 / (torch.sum(Q1, 0, keepdim=True))  # 列归一化Q

        if data_name == "Zaoyuan2":
            layers_count = 2   # Zaoyuan
        else:
            layers_count = 1   # other data

        # Spectra Transformation Sub-Network
        self.CNN_denoise = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(self.channel))
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i),
                                            nn.Conv2d(self.channel, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
            else:
                self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(128), )
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(128, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())

        # Spectra Transformation Sub-Network
        self.CNN_denoise2 = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.CNN_denoise2.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(self.channel))
                self.CNN_denoise2.add_module('CNN_denoise_Conv' + str(i),
                                            nn.Conv2d(self.channel, 128, kernel_size=(1, 1)))
                self.CNN_denoise2.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
            else:
                self.CNN_denoise2.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(128), )
                self.CNN_denoise2.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(128, 128, kernel_size=(1, 1)))
                self.CNN_denoise2.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())

        # postprocessing network
        self.CNN_denoise3 = nn.Sequential()
        self.CNN_denoise3.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(self.class_count))
        self.CNN_denoise3.add_module('CNN_denoise_Conv' + str(i),
                                    nn.Conv2d(self.class_count, self.class_count, kernel_size=(3, 3),padding=(1, 1)))
        self.CNN_denoise3.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())

        N1 = 2  # 输出卷积核数
        # input_channel = self.channel
        # input_channel = N1*self.channel
        input_channel = 128
        layers_count = 2
        # Superpixel-level Graph Sub-Network
        self.GCN_Branch1 = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.GCN_Branch1.add_module('GCN_Branch' + str(i), GCNLayerV2(input_channel, 128))
            elif (i>0) and (i < layers_count - 1):
                self.GCN_Branch1.add_module('GCN_Branch' + str(i), GCNLayer(128, 128))
            else:
                # self.GCN_Branch1.add_module('GCN_Branch' + str(i), GCNLayer(128, self.class_count))
                self.GCN_Branch1.add_module('GCN_Branch' + str(i), GCNLayerV2(128, 64))
                # self.GCN_Branch.add_module('GCN_Branch' + str(i), GCNLayer(128, 128))

        self.GCN_Branch2 = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.GCN_Branch2.add_module('GCN_Branch' + str(i), GCNLayerV2(input_channel, 128))
            elif (i>0) and (i < layers_count - 1):
                self.GCN_Branch2.add_module('GCN_Branch' + str(i), GCNLayer(128, 128))
            else:
                # self.GCN_Branch2.add_module('GCN_Branch' + str(i), GCNLayer(128, self.class_count))
                self.GCN_Branch2.add_module('GCN_Branch' + str(i), GCNLayerV2(128, 64))
                # self.GCN_Branch.add_module('GCN_Branch' + str(i), GCNLayer(128, 128))

        self.BN1 = nn.BatchNorm1d(128)
        self.GCN_Branch3 = GCNLayer(128, self.class_count)
        self.BN2 = nn.BatchNorm1d(128)
        # Softmax layer
        self.Softmax_linear = nn.Sequential(nn.Linear(128, self.class_count))
        self.Softmax_linear1 = nn.Sequential(nn.Linear(64, self.class_count))
        self.Softmax_linear2 = nn.Sequential(nn.Linear(64, self.class_count))
        self.Softmax_linear3 = nn.Sequential(nn.Linear(3*self.class_count, self.class_count))
        # self.Softmax_linear = nn.Sequential(nn.Linear(64, self.class_count))

        self.BN_H1 = nn.BatchNorm1d(self.channel)
        self.BN_H2 = nn.BatchNorm1d(self.channel)
        self.conv1d_H1 = nn.Conv1d(in_channels=1, out_channels=N1, kernel_size=10, stride=1,padding='same')
        self.conv1d_H2 = nn.Conv1d(in_channels=1, out_channels=N1, kernel_size=10, stride=1,padding='same')

    def forward(self, x: torch.Tensor):
        '''
        :param x: H*W*C
        :return: probability_map
        '''
        (h, w, c) = x.shape

        # 先去除噪声
        noise = self.CNN_denoise(torch.unsqueeze(x.permute([2, 0, 1]), 0))
        noise = torch.squeeze(noise, 0).permute([1, 2, 0])
        clean_x = noise  # 直连
        # clean_x = x  # 直连

        clean_x_flatten = clean_x.reshape([h * w, -1])

        x1 = torch.mm(self.norm_col_Q1.t(), clean_x_flatten)  # 低频部分
        x2 = clean_x_flatten


        H1 = x1
        for i in range(len(self.GCN_Branch1)):
            H1, _ = self.GCN_Branch1[i](H1, self.A1)
        H2 = x2
        for i in range(len(self.GCN_Branch2)):
            H2, _ = self.GCN_Branch2[i](H2, self.A2)

        # 使用矩阵乘法来计算 Posterior_PPSPG_ALL1
        Posterior_PPSPG_ALL1 = torch.matmul(self.Q1, H1)
        Posterior_PPSPG_ALL2 = H2
        # GCN_result = torch.matmul(self.Q, H)  # 这里self.norm_row_Q == self.Q

        # GCN_result = Posterior_PPSPG_ALL1*Posterior_PPSPG_ALL2  # 这里self.norm_row_Q == self.Q
        GCN_result = torch.cat([Posterior_PPSPG_ALL1, Posterior_PPSPG_ALL2], dim=-1)
        # GCN_result,_ = self.GCN_Branch2(H, self.A)

        # 两组特征融合(两种融合方式)
        # Y = torch.cat([GCN_result, CNN_result], dim=-1)
        Y0 = GCN_result
        Y0 = self.Softmax_linear(Y0)
        Y = Y0
        Y = F.softmax(Y, -1)
        return Y
