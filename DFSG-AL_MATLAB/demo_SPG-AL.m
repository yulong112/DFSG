%% v1,只更新原始特征MLR的训练样本，
%% 现象：还是会略微下降。说明：采样对原始特征MLR的影响较小，但还是需要调�?
%% 然后改成穷举，每轮采�?6个点，使目标函数�?��，目标函数是训练集上的交叉熵

clc
close all
clear all
% matlabpath(pathdef)
addpath(genpath('./MLR_toolbox'))
addpath(genpath('./Graph_toolbox'))
addpath(genpath('./sgwt_toolbox'));
addpath(genpath('./SuperPixel_toolbox'));
addpath(genpath('./Salinas-A'))
addpath(genpath('./zhengzhou'))
addpath(genpath('./Zaoyuan'))
addpath(genpath('./PaviaU'))
addpath(genpath('./Salinas'))
addpath(genpath('./Houston-Data/subsetV6'))
addpath(genpath('./LapYuHeSiDi'))

%% input param
sg_num=1;
mini_sp_num=15;
interval_sp_num=300;
% sp_num_vec=100:100:1500;
sp_num_vec=2400:200:3200;

            AL_method = 'MSSPG-singleAL';
kernel_h=15;

select_Data='IndianaPine';
select_Data='zhengzhou_20191110';
select_Data='LapYuHeSiDi';
% select_Data='SalinasA';
select_Data='Zaoyuan';
% select_Data='PaviaU';
% select_Data='PaviaU-subset';
% select_Data='SalinasR';
% select_Data='Houston-subsetV6';

%         TrainSamPerOrNum = [3,9];
%         TrainSamPerOrNum = [3,5,7,10,15,20,50,100];
%         TrainSamPerOrNum = [3,5,7,10];
%         TrainSamPerOrNum = [300];
%         TrainSamPerOrNum = [100];
%         TrainSamPerOrNum = [10,15,20,50];
        TrainSamPerOrNum = [5];

if strcmp(select_Data,'IndianaPine')
    superpix_allnum_vec=[[300,600];];
    %% IndianaPine 数据colormap
    mycolormap = [  0    0    0.5;           %背景�?
        %     1    0.75  0.75;                     %
        1    0.75    0;                      %深土�?
        1    0    0;                         %�?
        0    1    0;                         %�?
        1    0    1;                         %浅紫粉红�?
        0  0.5    1;                         %不深不浅蓝色
        %     0.75    0    0.8                        %
        0    1    1;                         %淡浅�?
        %     0  0.75  0.5;
        1    1    0.5;                       %浅米�?
        0.5 0.25    0;                       %�?
        0.5    0  0.5;                       %深紫�?
        0   0   0.8;                         %深蓝�?
        0.8   0.8   1;                       %灰色
        0.8   0.8   0;]                       %棕黄�?
    %% IndianaPine 数据
    load IndianaPine
    load IndianaPine_LabelMap
    HyperCube= IndianaPine;
    TruthMap = IndianaPine_LabelMap;
    clear IndianaPine IndianaPine_LabelMap
    [Height Width] = size(TruthMap);
    YStart = 1; % left-up point
    XStart = 1;
    YEnd = 145; % right-down point
    XEnd = 145;
    %DelBand = [1:5, 95, 96, 101:111,148:166,216:220];
    DelBand = [1:5, 101:111,148:166,216:220];
    DelClass = [1]; % 第一类�?是背景，类标�?
%     DelClass = [1 2 8 10 17]; % 第一类�?是背景，类标�?
    TrainPercent = 0.01;
    PreliminarySteps1
sigma_l=90;
weight_beta=0.9;
sp_num_vec=[2000];
% sp_num_vec=800:200:1400;
elseif strcmp(select_Data,'zhengzhou_20191110')
%% Zaoyuan ?1?7?1?7?1?7?1?7
load zhengzhou_20191110_data
load zhengzhou_20191110_label
HyperCube= zhengzhou_20191110_data;
TruthMap = zhengzhou_20191110_label;
clear zhengzhou_20191110_data zhengzhou_20191110_label
[Height Width] = size(TruthMap);

HyperCube=double(HyperCube);
HyperCube=fmapminmax(HyperCube);

YStart = 1; % left-up point
XStart = 1;
YEnd = Height; % right-down point
XEnd = Width;

DelBand = [];
DelClass = [1]; % ?1?7?1?7?0?5?1?7?1?7?1?7?1?7?1?7?0?3?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?0?2
TrainPercent = 0.01;
PreliminarySteps1
sigma_l=90;
weight_beta=0.9;
sp_num_vec=[4800];
% sp_num_vec=4400:400:5600;
superpix_allnum=[1500,1800];
superpix_num0=floor(Height*Width/60);

elseif strcmp(select_Data,'LapYuHeSiDi')
    load LapYuHeSiDi
    load LapYuHeSiDi_label
    HyperCube= LapYuHeSiDi;
    TruthMap = LapYuHeSiDi_label;
    clear LapYuHeSiDi LapYuHeSiDi_label
    [Height Width] = size(TruthMap);
    
    HyperCube=double(HyperCube);
    HyperCube=fmapminmax(HyperCube);
    TruthMap=double(TruthMap);
    
    YStart = 1; % left-up point
    XStart = 1;
    YEnd = Height; % right-down point
    XEnd = Width;
    
    DelBand = [];
    DelClass = [1]; % ?1?7?1?7?0?5?1?7?1?7?1?7?1?7?1?7?0?3?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?0?2
    TrainPercent = 0.01;
    PreliminarySteps1
    
    sp_num_vec=[3200];
%     sp_num_vec=2000:400:4800;
    sigma_l=160;
    weight_beta=0.2;
    
elseif strcmp(select_Data,'SalinasA')
    %% Salinas-A 数据
    load SalinasA
    % load SalinasA_corrected
    load SalinasA_gt
    HyperCube= salinasA;
    % HyperCube= salinasA_corrected;
    TruthMap = salinasA_gt;
    clear salinasA salinasA_gt
    [Height Width] = size(TruthMap);
    
    HyperCube=fmapminmax(HyperCube);
    
    YStart = 1; % left-up point
    XStart = 1;
    YEnd = 83; % right-down point
    XEnd = 86;
    
    DelBand = [];
    DelClass = [1]; % 第一类�?是背景，类标�?
    TrainPercent = 0.01;
    PreliminarySteps1
elseif strcmp(select_Data,'SalinasR')
    %% Salinas-R 数据
    load Salinas_corrected
    load Salinas_gt
    HyperCube= salinas_corrected;
    TruthMap = salinas_gt;
    clear salinas_corrected salinas_gt
    [Height Width] = size(TruthMap);
    
    HyperCube=fmapminmax(HyperCube);
    
    YStart = 81; % left-up point
    XStart = 33;
    YEnd = 282; % right-down point
    XEnd = 162;
    
    DelBand = [];
    DelClass = [1]; % 第一类�?是背景，类标�?
    TrainPercent = 0.01;
    PreliminarySteps1
elseif strcmp(select_Data,'Zaoyuan')
    superpix_allnum_vec=[[900,1500];];
    %% Zaoyuan 数据colormap
    mycolormap = [  0    0    0.5;           %背景�?
        1    0.75    0;                      %深土�?
        1    0    0;                         %�?
        0    1    0;                         %�?
        1    0    1;                         %浅紫粉红�?
        0.8   0.8   1;                       %灰色
        0    1    1;                         %淡浅�?
        1    1    0.5;                       %浅米�?
        0.5 0.25    0;]                      %�?
    %% Zaoyuan 数据
%     load HyperCube_OMIS4Class2
%     % load SalinasA_corrected
%     load TruthMap2
%     HyperCube= HyperCube_OMIS4Class2;
%     % HyperCube= salinasA_corrected;
%     TruthMap = TruthMap2;
%     clear HyperCube_OMIS4Class2 TruthMap2
%     [Height Width] = size(TruthMap);
    
    addpath(genpath('../HyperImage_data/Zaoyuan'))
%     load HyperCube_OMIS4Class2
%     % load SalinasA_corrected
%     load TruthMap2
%     HyperCube= HyperCube_OMIS4Class2;
%     % HyperCube= salinasA_corrected;
%     TruthMap = TruthMap2;
%     clear HyperCube_OMIS4Class2 TruthMap2
    load Zaoyuan
    load Zaoyuan_gt
    HyperCube= Zaoyuan;
    TruthMap = Zaoyuan_gt;
    clear Zaoyuan Zaoyuan_gt
    [Height Width Bands] = size(HyperCube);
    rmpath(genpath('../HyperImage_data/Zaoyuan'));
    
    YStart = 1; % left-up point
    XStart = 1;
    YEnd = 137; % right-down point
    XEnd = 202;
    
    DelBand = [];
    DelClass = [1]; % 第一类�?是背景，类标�?
    TrainPercent = 0.01;
    PreliminarySteps1
sigma_l=100;
weight_beta=0.8;
sp_num_vec=[2400];
elseif strcmp(select_Data,'PaviaU')
    superpix_allnum_vec=[[300,900];];
    %% PaviaU 数据
    load PaviaU
    load PaviaU_gt
    HyperCube= paviaU;
    TruthMap = paviaU_gt;
    clear paviaU paviaU_gt
    [Height Width] = size(TruthMap);
    HyperCube=fmapminmax(HyperCube);
    
    % YStart = 158; % left-up point
    % XStart = 1;
    % YEnd = 353; % right-down point
    % XEnd = 182;
    
    %     YStart = 158; % left-up point
    %     XStart = 1;
    %     YEnd = 357; % right-down point
    %     XEnd = 186;

    YStart = 1; % left-up point
    XStart = 1;
    YEnd = Height; % right-down point
    XEnd = Width;
    
%     YStart = 210; % left-up point
%     XStart = 57;
%     YEnd = 413; % right-down point
%     XEnd = 206;
    
    DelBand = [];
    DelClass = [1]; % 第一类�?是背景，类标�?
    TrainPercent = 0.01;
    PreliminarySteps1
sigma_l=18;
weight_beta=0.1;
sp_num_vec=[2400];
elseif strcmp(select_Data,'PaviaU-subset')
    superpix_allnum_vec=[[300,900];];
    %% PaviaU 数据
    load PaviaU
    load PaviaU_gt
    HyperCube= paviaU;
    TruthMap = paviaU_gt;
    clear paviaU paviaU_gt
    [Height Width] = size(TruthMap);
    HyperCube=fmapminmax(HyperCube);
    
    % YStart = 158; % left-up point
    % XStart = 1;
    % YEnd = 353; % right-down point
    % XEnd = 182;
    
    %     YStart = 158; % left-up point
    %     XStart = 1;
    %     YEnd = 357; % right-down point
    %     XEnd = 186;

    YStart = 210; % left-up point
    XStart = 57;
    YEnd = 413; % right-down point
    XEnd = 206;
    
    DelBand = [];
    DelClass = [1]; % 第一类�?是背景，类标�?
    TrainPercent = 0.01;
    PreliminarySteps1
sigma_l=18;
weight_beta=0.1;
elseif strcmp(select_Data,'Houston-subsetV6')
    %% Houston-subsetV5 数据
    load NCALM_Houston_subV6
    load Houston_subV6_GT
    load Houston_subV6_TrainTestSet
    load Houston_subV6_MyColorMap
    YStart = 1; % left-up point
    XStart = 1;
    YEnd = size(TruthMap,1); % right-down point
    XEnd = size(TruthMap,2);
    DelBand = [];
    DelClass = [1]; % 第一类�?是背景，类标�?
    TrainPercent = 0.01;
    PreliminarySteps1
    
    [Height Width] = size(TruthMap);
    
    for i = 1:length(ClassLabel)
        TrainSamNumAClass(i,1) = length(find(train_setV6(2,:) == ClassLabel(i)));
    end
sigma_l=18;
weight_beta=0.5;
end
[Height Width] = size(TruthMap);
no_classes=length(TotalSamNumAClass)

%% �ο���ǩ����
TruthMap2=TruthMap;
% changeclass=DelClass(DelClass~=1);
% [~,change_pos]=ismember(TruthMap2,changeclass);
% TruthMap2(logical(change_pos))=0;
for i=2:length(DelClass)
    TruthMap2(TruthMap2==(DelClass(i)-1))=0;
end
%% label rearranging
TruthMap1D = TruthMap2(:);
UniqueLabel = unique(TruthMap1D);
UniqueLabel = sort(UniqueLabel, 'ascend');
for i = 1: length(UniqueLabel)
    TruthMap1D( find(TruthMap1D==UniqueLabel(i)) ) = i-1;
end
TruthMap2 = reshape(TruthMap1D, size(TruthMap2));
TruthMap1D=TruthMap2(:);

AccRate_all={};
OA_std_all={};
AverageAcc_all={};
AA_std_all={};
IndiAccRate_all={};
kappaCoef_all={};
zscore_ESD_all={};
kappaCoef_std_all={};
    Time_crossValidation_all={};
    Time_classification_all={};
    Time_computeWeight_all={};
    
for iter_sp_num=1:length(sp_num_vec)
    superpix_num=sp_num_vec(iter_sp_num)
    
    superpix_img = superpixel_cut(HyperCube,superpix_num);
    sp_num=max(superpix_img);
    
    
    SP_Gabor_data=[];
    SP_Gabor_data2=[];
    SP_Gabor_data3=[];
    SP_Gabor_data4=[];
%     HyperCube2 = Gaussian_GaussianDerivative(HyperCube);
%     
%     [Height,Width,Bands] = size(HyperCube2);
%     HyperCube2_vec = reshape(HyperCube2, Width*Height, Bands); %3D�?2D
%     HyperCube2DR_vec = pca_row(HyperCube2_vec,30);
    [Height,Width,Bands] = size(HyperCube);
    HyperCube_vec = reshape(HyperCube, Width*Height, Bands); %3D�?2D
    HyperCubeDR_vec = pca_row(HyperCube_vec,30);
    
    IndiAccRate=[];
    AA_std=[];
    kappaCoef=[];
    kappaCoef_std=[];
    zscore_ESD=[];
    OA_std=[];
    AccRate=[];
    Time_classification=[];
    Time_crossValidation=[];
    Time_computeWeight=[];
    
    split_flags=ismember('Houston-subset',select_Data);
    split_flags=length(find(split_flags==0));%split_flags=0, is Houston-subset; split_flags>0, is not Houston-subset;
    % TrainSamPerOrNum = [0.02 0.03 0.04 0.05];
    % TrainSamPerOrNum = [2];
    if split_flags==0
%         TrainSamPerOrNum = [100];
        TrainSamPerOrNum = [3,5,7,9];
    else
    end
    FlagOfPerOrNum = 1; %the sign of setting the number of the training sample with the percent or sample number. 0 for Percent; 1 for Number;
    for PerNumNo = 1: length(TrainSamPerOrNum)
        TrainPerNum = TrainSamPerOrNum(PerNumNo);
        if FlagOfPerOrNum == 0
            NumClassSam = ceil(TrainPerNum*TotalSamNumAClass);
        else
            NumClassSam = ceil(TrainPerNum*ones(length(TotalSamNumAClass),1));
            NumClassSam(NumClassSam >= TotalSamNumAClass) = ceil(0.7*TotalSamNumAClass(NumClassSam >= TotalSamNumAClass));
        end
        
        NeighborMask = [1];
        
        AccRate_RandNo = [];
        IndiAccRate_RandNo = [];
        kappaCoef_RandNo = [];
        zscore_ESD_RandNo = [];
        
        Time_classification_RandNo = [];
        Time_crossValidation_RandNo = [];
        Time_weight_construct_RandNo = [];
        
        RandNo = 1:5;%Zaoyuan:3,4,6,7,8,min:8
        %                     RandNo = [1,100,500,1000,2000,3899];
%             RandNo = [4,5,8,13,20];
        for RandNo_i=RandNo
            if split_flags==0
%                 NumClassSam = ceil(TrainPerNum*TrainSamNumAClass);
                [NormTrainSam, NormTestSam, TrainLabels, TestLabels, TrainSamLoc, TestSamLoc] = fgetTrainTestSam_Houston...
                    (HyperCube,train_setV6,test_setV6,NumClassSam, RandNo_i);
            else
                [NormTrainSam, NormTestSam, TrainLabels, TestLabels, TrainSamLoc, TestSamLoc] = fSetTrainTestSam_Neighbor_V2...
                    (HyperCube, TruthMap, SelClassNo, NeighborMask, RandNo_i, NumClassSam);
            end
            
            NormTrainSam = NormTrainSam{1};
            NormTestSam = NormTestSam{1};
            
%             matSaveDir = ['./Result/',select_Data,'/',AL_method,'V2.6/initTrainNum-', num2str(TrainPerNum),'/RngSeed-', num2str(RandNo_i)];
% %             matSaveDir = ['./Result/',select_Data,'/',AL_method,'V2.6_discuss_10iter4_noupdate/initTrainNum-', num2str(TrainPerNum),'/RngSeed-', num2str(RandNo_i)];
% %             matSaveDir = ['./Result/',select_Data,'/CASSL-BT-MLR-V3/RngSeed-', num2str(RandNo_i)];
% %             matSaveDir = ['./Result/',select_Data,'/CNN-AL-MRF/RngSeed-', num2str(RandNo_i)];
%             load([matSaveDir,'/TrainLabels.mat'],'TrainLabels')
%             load([matSaveDir,'/TestLabels.mat'],'TestLabels')
%             load([matSaveDir,'/TrainSamLoc.mat'],'TrainSamLoc')
%             load([matSaveDir,'/TestSamLoc.mat'],'TestSamLoc')
            
            InitNormTrainSam=NormTrainSam;
            InitTrainSamLoc=TrainSamLoc;
            InitTrainLabels=TrainLabels;
            AvailableLoc=[TrainSamLoc;TestSamLoc];
            
                    %% generate data for transformer
%                     TR = zeros(size(TruthMap2,1)*size(TruthMap2,2),1);
%                     TR(TrainSamLoc) = TrainLabels;
%                     TR = reshape(TR,size(TruthMap2,1),size(TruthMap2,2));
%                     figure;imagesc(TR)
%                     TE = zeros(size(TruthMap2,1)*size(TruthMap2,2),1);
%                     TE(TestSamLoc) = TestLabels;
%                     TE = reshape(TE,size(TruthMap2,1),size(TruthMap2,2));
%                     figure;imagesc(TE)
%                     input = HyperCube;
% %                     save(['Indian_', num2str(TrainPerNum),'_', num2str(RandNo_i), '.mat'],'input','TR','TE');
%                     save(['./Result/',select_Data,'_', num2str(TrainPerNum),'_', num2str(RandNo_i), '.mat'],'input','TR','TE');
%                     [hh, ww] = size(TR);
%                     TR1D = reshape(TR, hh*ww, 1).';
%                     TR_ind = find(TR1D>0);
%                     
%                     [hh, ww] = size(TE);
%                     TE1D = reshape(TE, hh*ww, 1).';
%                     TE_ind = find(TE1D>0);
%                     
%                     save(['./Result/TR_ind_s',num2str(TrainPerNum),'_iter',num2str(RandNo_i),'.mat'],'TR_ind');
%                     save(['./Result/TE_ind_s',num2str(TrainPerNum),'_iter',num2str(RandNo_i),'.mat'],'TE_ind');
% 
%                     train_gt = TR;
%                     test_gt = TE;
% %                     save(['./trainTestSplit/',select_Data,'/sample', num2str(TrainPerNum), '_run', num2str(RandNo_i), '.mat'],'train_gt','test_gt');
                    
            initial_Labels = zeros(sp_num,no_classes);
            TrainSamLoc_sp = [];
            TrainLabels_sp = [];
            for select_i=1:length(TrainSamLoc)
                sp_idx=superpix_img(TrainSamLoc(select_i));
                initial_Labels(sp_idx,TrainLabels(select_i)) = 1;
                TrainSamLoc_sp = [TrainSamLoc_sp;sp_idx];
                TrainLabels_sp = [TrainLabels_sp;TrainLabels(select_i)];
            end
            initial_Labels=sparse(initial_Labels);
            TrainSamLoc_sp = double(TrainSamLoc_sp);
            
            % 设定候�?集，只需要所有的测试样本和测试样本位�?
            CandiSamLoc=TestSamLoc;
            CandiLabels=TestLabels;
            NormCandiSam=NormTestSam;

            %% 超像素谱图聚类�?
            Time_start=clock;
            Time_weight_construct=[];
            one_accuracy=[];
            para_alpha=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.93,0.95,0.97,0.99];
            Time_weight_start=clock;
            WeightMat = preprocess_sp_data_otherpaper(HyperCube,superpix_img, 0, sigma_l, kernel_h, weight_beta);
            N = size(WeightMat,1);
            d = sum(WeightMat,2);
            Dinv = spdiags(d,0,N,N);
            Laplace_Mat_no_normalize=Dinv-WeightMat;

            time_tmp1 = etime(clock,Time_weight_start)
            Time_weight_construct=[Time_weight_construct time_tmp1];

            for iter_alpha=1:length(para_alpha)
                [~,PredLabels2] = SemiSupervised_Graph_Classifier_v2(Laplace_Mat_no_normalize, initial_Labels, [], para_alpha(iter_alpha));
                PredLabels_ALL=zeros(Width*Height,1);
                for select_i=1:sp_num
                    pos_idx=find(superpix_img==select_i);
                    PredLabels_ALL(pos_idx)=PredLabels2(select_i);
                end
                PredLabels2 = PredLabels_ALL(TestSamLoc);
                one_accuracy(iter_alpha) = fAccuracy(PredLabels2,TestLabels,0);% 每个选择的尺度上的测试样本准确率
            end
            [maxAcc1,bestalphaNo] =max(one_accuracy);
            bestalpha=para_alpha(bestalphaNo);

            time_tmp = etime(clock,Time_start)
            Time_crossValidation_RandNo = [Time_crossValidation_RandNo time_tmp];
            Time_weight_construct_RandNo = [Time_weight_construct_RandNo mean(Time_weight_construct)];
            Time_start=clock;

            [~,PredLabels2,Posterior_PPSPG] = SemiSupervised_Graph_Classifier_v2(Laplace_Mat_no_normalize, initial_Labels, [], bestalpha);
            PredLabels_ALL=zeros(Width*Height,1);
            Posterior_PPSPG_ALL=zeros(Width*Height,no_classes);
            for select_i=1:sp_num
                pos_idx=find(superpix_img==select_i);
                PredLabels_ALL(pos_idx)=PredLabels2(select_i);
                Posterior_PPSPG_ALL(pos_idx,:)=repmat(Posterior_PPSPG(select_i,:),length(pos_idx),1);
            end
            PredLabels_PPSPG = PredLabels_ALL(TestSamLoc);
            confidence_PPSPG_ALL = max(Posterior_PPSPG_ALL,[],2);

            time_tmp = etime(clock,Time_start)
            Time_classification_RandNo = [Time_classification_RandNo time_tmp];

            tmp_max_OA=[];
            tmp_max_OA(1) = fAccuracy(PredLabels_PPSPG,TestLabels,0)

            
            initTrainSamLoc = TrainSamLoc;
            initTrainLabels = TrainLabels;
            total_iter=0;
            for iter=1:total_iter
%                     [PredLabels,posterior]=MLR_Classifier_fun(HyperCube,TrainSamLoc,TrainLabels,NormCandiSam);
                %                     AccRate_MLR_prev = fAccuracy(PredLabels,TestLabels,0)

                para_alpha=0.5;
                [~,PredLabels2,Posterior_PPSPG] = SemiSupervised_Graph_Classifier_v2(Laplace_Mat_no_normalize, initial_Labels, [], para_alpha);
                PredLabels_ALL=zeros(Width*Height,1);
                Posterior_PPSPG_ALL=zeros(Width*Height,no_classes);
                for select_i=1:sp_num
                    pos_idx=find(superpix_img==select_i);
                    PredLabels_ALL(pos_idx)=PredLabels2(select_i);
                    Posterior_PPSPG_ALL(pos_idx,:)=repmat(Posterior_PPSPG(select_i,:),length(pos_idx),1);
                end
                PredLabels_PPSPG_candi = PredLabels_ALL(CandiSamLoc);
                confidence_PPSPG_ALL = max(Posterior_PPSPG_ALL,[],2);

                BT_num = 2;
                if iter==total_iter
                    BT_num = 2;
                end
                % ����ѧϰBT����
                sp_ind_candi = unique(superpix_img(CandiSamLoc));
                pactive = Posterior_PPSPG(sp_ind_candi,:)';
                pactive = sort(pactive,'descend');
                pactive_minBT_PPSPG = pactive(1,:)-pactive(2,:);
                BT_scores = pactive_minBT_PPSPG;
                [sortmminppBT, indexsortminppBT] = sort(BT_scores);
                PPSPG_idx_sel = indexsortminppBT(1:BT_num);
                PPSPG_sel_sp_ind_candi = sp_ind_candi(PPSPG_idx_sel);
                        candi_pix_locs_sel = [];
                for process_i = 1:length(PPSPG_sel_sp_ind_candi)
                    candi_spi = PPSPG_sel_sp_ind_candi(process_i);
                    candi_pix_locs = find(superpix_img==candi_spi);
                    [~,candi_pix_locs_id]=ismember(candi_pix_locs,CandiSamLoc);
                    candi_pix_locs_id = candi_pix_locs_id(candi_pix_locs_id~=0);
                    candi_pix_locs = CandiSamLoc(candi_pix_locs_id);
                    
                        candi_pix_locs_sel = [candi_pix_locs_sel;candi_pix_locs(1)];
                end

                % 更新训练样本
                [~,cfd_match_idx_sel] = ismember(candi_pix_locs_sel,CandiSamLoc);
                    cfd_match_idx_sel = cfd_match_idx_sel(cfd_match_idx_sel~=0);
                NewTrainSamLoc = CandiSamLoc(cfd_match_idx_sel);
%                 NewTrainLabels = PredLabels_MLR1_candi(cfd_match_idx_sel);
                NewTrainLabels = CandiLabels(cfd_match_idx_sel);
                NewNormTrainSam = NormCandiSam(cfd_match_idx_sel,:);

                TrainSamLoc = [TrainSamLoc;NewTrainSamLoc];
                TrainLabels = [TrainLabels;NewTrainLabels];
                NormTrainSam = [NormTrainSam;NewNormTrainSam];

                % 将�?选集排除掉一�?
                CandiSamLoc(cfd_match_idx_sel)=[];
                CandiLabels(cfd_match_idx_sel)=[];
                NormCandiSam(cfd_match_idx_sel,:)=[];
                TestSamLoc(cfd_match_idx_sel)=[];
                TestLabels(cfd_match_idx_sel)=[];

                % 更新超像素分�?
                for select_i=1:sp_num
                    pos_idx=find(superpix_img==select_i);
                    spTrainflag=ismember(TrainSamLoc,pos_idx);
                    spTrainSamLoc = TrainSamLoc(spTrainflag);
                    spTrainLabels = TrainLabels(spTrainflag);
                    if length(spTrainSamLoc)
                        spTrainLabels=yl_label_reranging(spTrainLabels)+1;
                        if max(spTrainLabels)>1
                            % 超过两类，分割超像素为多�?
                            WeightMat_sp = SNG_sp_patch(HyperCube_vec(pos_idx,:), pos_idx, Height,Width);
                            N_sp = size(WeightMat_sp,1);
                            d = sum(WeightMat_sp,2);
                            d(d~=0) = d(d~=0).^(-1/2);
                            Dinv_sp = spdiags(d,0,N_sp,N_sp);
                            SW_sp = Dinv_sp*WeightMat_sp*Dinv_sp;
                            initial_Labels_sp = zeros(N_sp,max(spTrainLabels));
                            for spTr_i=1:length(spTrainSamLoc)
                                initial_Labels_sp(spTr_i,spTrainLabels(spTr_i)) = 1;
                            end
                            initial_Labels_sp=sparse(initial_Labels_sp);
                            para_alpha=0.5;
                            UU = speye(N_sp) - para_alpha * SW_sp;
                            F = UU \ initial_Labels_sp; % classification function F = (I - \alpha S)^{-1}Y
                            [~, PredLabels_sp] = max(F, [], 2); %simply checking which of elements is largest in each row
                            % 将其中的某些超像素序号增�?
                            for splabel_i=2:max(PredLabels_sp)
                                sploc_i = find(PredLabels_sp==splabel_i);
                                max_sp_ind = max(superpix_img);
                                superpix_img(pos_idx(sploc_i))=max_sp_ind+1;
                            end
                        end
                    end
                end
                
                % 更新图的训练样本
                sp_num=max(superpix_img);
                initial_Labels = zeros(sp_num,no_classes);
                TrainSamLoc_sp = [];
                TrainLabels_sp = [];
                for select_i=1:length(TrainSamLoc)
                    sp_idx=superpix_img(TrainSamLoc(select_i));
                    initial_Labels(sp_idx,TrainLabels(select_i)) = 1;
                    TrainSamLoc_sp = [TrainSamLoc_sp;sp_idx];
                    TrainLabels_sp = [TrainLabels_sp;TrainLabels(select_i)];
                end
                initial_Labels=sparse(initial_Labels);
                TrainSamLoc_sp = double(TrainSamLoc_sp);

                
                % 更新图结�?
                WeightMat = preprocess_sp_data_otherpaper(HyperCube,superpix_img, 0, sigma_l, kernel_h, weight_beta);
                N = size(WeightMat,1);
                d = sum(WeightMat,2);
                Dinv = spdiags(d,0,N,N);
                Laplace_Mat_no_normalize=Dinv-WeightMat;

                %谱图聚类�?
                para_alpha=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.93,0.95,0.97,0.99];
                for iter_alpha=1:length(para_alpha)
                    [~,PredLabels2] = SemiSupervised_Graph_Classifier_v2(Laplace_Mat_no_normalize, initial_Labels, [], para_alpha(iter_alpha));
                    PredLabels_ALL=zeros(Width*Height,1);
                    for select_i=1:sp_num
                        pos_idx=find(superpix_img==select_i);
                        PredLabels_ALL(pos_idx)=PredLabels2(select_i);
                    end
                    PredLabels2 = PredLabels_ALL(TestSamLoc);
                    one_accuracy(iter_alpha) = fAccuracy(PredLabels2,TestLabels,0);% 每个选择的尺度上的测试样本准确率
                end
                [maxAcc1,bestalphaNo] =max(one_accuracy);
                bestalpha=para_alpha(bestalphaNo);
                [~,PredLabels2] = SemiSupervised_Graph_Classifier_v2(Laplace_Mat_no_normalize, initial_Labels, [], bestalpha);
                PredLabels_ALL=zeros(Width*Height,1);
                for select_i=1:sp_num
                    pos_idx=find(superpix_img==select_i);
                    PredLabels_ALL(pos_idx)=PredLabels2(select_i);
                end
                PredLabels_PPSPG = PredLabels_ALL(TestSamLoc);
                tmp_max_OA(iter+1) = fAccuracy(PredLabels_PPSPG,TestLabels,0)

            end
            
            
            AccRate_RandNo = [AccRate_RandNo tmp_max_OA(end)];

            IndiAccRate_RandNo = [IndiAccRate_RandNo fAccuracy(PredLabels_PPSPG,TestLabels,1)];
            [kappaCoef_tmp, zscore_ESD_tmp] = fKappaCoef(PredLabels_PPSPG,TestLabels);
            kappaCoef_RandNo = [kappaCoef_RandNo kappaCoef_tmp];
            zscore_ESD_RandNo = [zscore_ESD_RandNo zscore_ESD_tmp];
        end
        AccRate(:,PerNumNo) = mean(AccRate_RandNo,2)
        OA_std(:,PerNumNo) = std(AccRate_RandNo,0,2)
        IndiAccRate(:,PerNumNo) = mean(IndiAccRate_RandNo,2)
        %                     AccRate(:,PerNumNo) = max(one_accuracy,[],2)
        AA_std(:,PerNumNo) = std(mean(IndiAccRate_RandNo,1))
        
        kappaCoef(:,PerNumNo) = mean(kappaCoef_RandNo,2)
        kappaCoef_std(:,PerNumNo) = std(kappaCoef_RandNo,0,2)
        zscore_ESD(:,PerNumNo) = mean(zscore_ESD_RandNo,2);
        
        Time_crossValidation(:,PerNumNo) = mean(Time_crossValidation_RandNo,2)
        Time_classification(:,PerNumNo) = mean(Time_classification_RandNo,2)
        
        Time_computeWeight(:,PerNumNo) = mean(Time_weight_construct_RandNo,2);
                        
    end
    AccRate_all{iter_sp_num}=AccRate;
    OA_std_all{iter_sp_num}=OA_std;
    AverageAcc=mean(IndiAccRate,1)
    AverageAcc_all{iter_sp_num} = AverageAcc;
    AA_std_all{iter_sp_num}=AA_std;
    IndiAccRate_all{iter_sp_num} = IndiAccRate;
    kappaCoef_all{iter_sp_num} = kappaCoef;
    kappaCoef_std_all{iter_sp_num} = kappaCoef_std;
    zscore_ESD_all{iter_sp_num} = zscore_ESD;
    
    Time_crossValidation_all{iter_sp_num} = Time_crossValidation;
    Time_classification_all{iter_sp_num} = Time_classification;
    Time_computeWeight_all{iter_sp_num} = Time_computeWeight;
end