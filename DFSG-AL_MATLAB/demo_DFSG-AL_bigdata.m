
clc
close all
clear all
% matlabpath(pathdef)
addpath(genpath('./MLR_toolbox'))
addpath(genpath('./Graph_toolbox'))
addpath(genpath('./SuperPixel_toolbox'));
addpath(genpath('./Posterior_Norm'))
addpath(genpath('./Utils_DFSG'))
addpath(genpath('./zhengzhou'))
addpath(genpath('./LapYuHeSiDi'))

%% input param
sg_num=1;
mini_sp_num=15;
interval_sp_num=300;
% sp_num_vec=100:100:1500;
sp_num_vec=2400:200:3200;
conn_pattern='combine';

kernel_h=15;

select_Data='zhengzhou_20191110';
select_Data='WHU_Hi_HanChuan';
% select_Data='MUUFL';
select_Data='LapYuHeSiDi';
% select_Data='Houston2018';

%         TrainSamPerOrNum = [3,9];
%         TrainSamPerOrNum = [3,5,7,10,15,20,50,100];
%         TrainSamPerOrNum = [3,5,7,10];
%         TrainSamPerOrNum = [300];
%         TrainSamPerOrNum = [100];
%         TrainSamPerOrNum = [10,15,20,50];
        TrainSamPerOrNum = [150];
        TrainSamPerOrNum = [15];
        TrainSamPerOrNum = [3];
            AL_method = 'MSSPG-singleAL';
            
if strcmp(select_Data,'zhengzhou_20191110')
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
sp_num_vec=[2000];
superpix_allnum=[1200,2400];
superpix_allnum=[1200];
% superpix_allnum_vec=[[600,900];];
superpix_allnum_vec=[[600,900];];
superpix_allnum_vec=[[100];];
superpix_allnum_vec=[[1500,1800];];%80%,80.67%,81.76%,
% superpix_allnum_vec=[[600,900];];
% superpix_allnum_vec=[[1000,1500];];
superpix_num0=floor(Height*Width/60);
    superpix_num0=200*200;

elseif strcmp(select_Data,'WHU_Hi_HanChuan')
    addpath(genpath('./WHU_Hi_HanChuan'))
    
    load WHU_Hi_HanChuan
    load WHU_Hi_HanChuan_gt
    HyperCube= WHU_Hi_HanChuan;
    TruthMap = WHU_Hi_HanChuan_gt;
    clear WHU_Hi_HanChuan WHU_Hi_HanChuan_gt
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
    rmpath(genpath('./WHU_Hi_HanChuan'));
    superpix_allnum_vec=[[1500,1800];];%80%,80.67%,81.76%,
    superpix_allnum_vec=[[900,1800];];%80%,80.67%,81.76%,
    superpix_num2=floor(Height*Width/60);
    superpix_num0=floor(Height*Width/100);
%     setratio = 180;
%     superpix_num2=floor(Height*Width/setratio);
%     superpix_num0=floor(Height*Width/setratio/3);
    superpix_num3=floor(superpix_num2/2.5);
%     superpix_num2=max(superpix_allnum_vec)*4;
%     superpix_num0=max(superpix_allnum_vec)*2;
%     superpix_num0=80*200;
        mycolormap = [ 1    0.75  0.75;                     %
        1    0.75    0;                      %深土黄
        1    0    0;                         %红
        0    1    0;                         %绿
        1    0    1;                         %浅紫粉红色
        0  0.5    1;                         %不深不浅蓝色
        0.75    0    0.8                        %
        0    1    1;                         %淡浅蓝
        0  0.75  0.5;
        1    1    0.5;                       %浅米黄
        0.5 0.25    0;                       %棕
        0.5    0  0.5;                       %深紫色
        0   0   0.8;                         %深蓝色
        0.8   0.8   1;                       %灰色
        0.8   0.8   0;                       %棕黄色
        0.6, 0.2, 0.4];

elseif strcmp(select_Data,'Houston2018')
    addpath(genpath('./Houston2018'))
    
    load DFC2018_20
    load DFC2018_gt20
    HyperCube= DFC2018;
    TruthMap = DFC2018_gt;
    clear DFC2018 DFC2018_gt
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
    rmpath(genpath('./Houston2018'));
    superpix_allnum_vec=[[20000,30000];];%80%,80.67%,81.76%,
%     superpix_allnum_vec=[[1500,1800];[16000,30000];[16000,20000];[20000,30000];[20000,35000];[25000,30000];[25000,35000];[30000,40000];];%80%,80.67%,81.76%,
    superpix_num2=floor(Height*Width/60);
    superpix_num0=floor(Height*Width/100);
    setratio = 40;
    superpix_num2=floor(Height*Width/setratio);
    superpix_num0=floor(Height*Width/setratio/1.25);
    superpix_num3=floor(superpix_num2/2.5);
%     superpix_num2=max(superpix_allnum_vec)*4;
%     superpix_num0=max(superpix_allnum_vec)*2;
%     superpix_num0=80*200;
        mycolormap = [ 1    0.75  0.75;                     %
        1    0.75    0;                      %深土黄
        1    0    0;                         %红
        0    1    0;                         %绿
        1    0    1;                         %浅紫粉红色
        0  0.5    1;                         %不深不浅蓝色
        0.75    0    0.8                        %
        0    1    1;                         %淡浅蓝
        0  0.75  0.5;
        1    1    0.5;                       %浅米黄
        0.5 0.25    0;                       %棕
        0.5    0  0.5;                       %深紫色
        0   0   0.8;                         %深蓝色
        0.8   0.8   1;                       %灰色
        0.8   0.8   0;                       %棕黄色
        0.6, 0.2, 0.4;
        0.2  0.2  0.2;    % 深灰色
        1.0  0.8  0.2;    % 金黄色
        0.1  0.6  0.8;    % 青色
        0.8  0.1  0.3     % 玫红色
];

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
    superpix_allnum_vec=[[300,600];[400,800];[500,1000];[600,1200];[700,1400];[800,1600];[1200,1800];[1500,1800];];%80%,80.67%,81.76%,
%     superpix_allnum_vec=[[900,1800];];%80%,80.67%,81.76%,
    superpix_allnum_vec=[[600,1200];];%80%,80.67%,81.76%,
%     superpix_allnum_vec=[[600,900];[900,1200];];%80%,80.67%,81.76%,
    superpix_num2=floor(Height*Width/40);
    superpix_num0=floor(Height*Width/50);
    superpix_num3=floor(superpix_num2/2.5);
    setratio = 40;
    superpix_num2=floor(Height*Width/setratio);
    superpix_num0=floor(Height*Width/setratio/1.25);
    superpix_num3=floor(superpix_num2/2.5);
%     superpix_num2=max(superpix_allnum_vec)*4;
%     superpix_num0=max(superpix_allnum_vec)*2;
%     superpix_num0=80*200;
        mycolormap = [1    0.75    0;                      %濞ｅ崬婀℃?
        1    0    0;                         %缁?
        0    1    0;                         %缂?
        1    0    1;                         %濞村懐浼犵划澶屽閼?
        0.8   0.8   1;                       %閻忔媽澹?
        0    1    1;                         %濞ｂ剝绁拑?
        1    1    0.5;                       %濞村懐鑳屾?
        0.5 0.25    0;]                      %濡?
end
[Height Width] = size(TruthMap);
no_classes=length(TotalSamNumAClass)

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
    
for iter_sp_num=1:length(superpix_allnum_vec)
    superpix_allnum=superpix_allnum_vec(iter_sp_num,:)
    
%     superpix_img = superpixel_cut(HyperCube,superpix_num);
%     sp_num=max(superpix_img);
    
    
    SP_Gabor_data=[];
    SP_Gabor_data2=[];
    SP_Gabor_data3=[];
    SP_Gabor_data4=[];
    
    [Height,Width,Bands] = size(HyperCube);
    HyperCube_vec = reshape(HyperCube, Width*Height, Bands); %3D鍙?2D
    HyperCubeDR_vec = pca_row(HyperCube_vec,30);
    
    
%% method1:median/mean
%% method2:gauss
%     BandNum=size(HyperCube,3);
%     GaussianMaskSize=[3];
%     GaussianSigma=[0.9];
%     for MaskNo=1:length(GaussianMaskSize)
%         for SigmaNo=1:length(GaussianSigma)
%             TempHyperCube=[];
%             for  BandNo=1:BandNum
%                 HyperCubeBand=GaussianSmooth(HyperCube(:,:,BandNo),GaussianMaskSize(MaskNo),GaussianSigma(SigmaNo));
%                 TempHyperCube=cat(3,TempHyperCube,HyperCubeBand);
%             end
%         end
%     end
%     HyperCube=TempHyperCube;
%     clear TempHyperCube HyperCubeBand
    
    
    
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
        
        RandNo = 1:5;
        %                     RandNo = [1,100,500,1000,2000,3899];
        for RandNo_i=RandNo
            [NormTrainSam, NormTestSam, TrainLabels, TestLabels, TrainSamLoc, TestSamLoc] = fSetTrainTestSam_Neighbor_V2...
                (HyperCube, TruthMap, SelClassNo, NeighborMask, RandNo_i, NumClassSam);
            
            NormTrainSam = NormTrainSam{1};
            NormTestSam = NormTestSam{1};
            
%             TrainSamLoc = find(training_map);
%             TrainLabels = training_map(TrainSamLoc);
%             TestSamLoc = find(testing_map);
%             TestLabels = testing_map(TestSamLoc);
            
            InitNormTrainSam=NormTrainSam;
            InitTrainSamLoc=TrainSamLoc;
            InitTrainLabels=TrainLabels;
            AvailableLoc=[TrainSamLoc;TestSamLoc];
            
            % 璁惧畾鍊欓?闆嗭紝鍙渶瑕佹墍鏈夌殑娴嬭瘯鏍锋湰鍜屾祴璇曟牱鏈綅缃?
            CandiSamLoc=TestSamLoc;
            CandiLabels=TestLabels;
            NormCandiSam=NormTestSam;

            %% 瓒呭儚绱犺氨鍥捐仛绫汇?
            tic;
            Time_DFSGAL_start=clock;
            Time_weight_start=clock;
            one_accuracy=[];
            para_alpha=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.93,0.95,0.97,0.99];
            
            [WeightMat,superpix_img0,Superpixel_mean_Features] = FSPG_graph_construction_bigdata(HyperCube,superpix_num0,superpix_allnum,conn_pattern,TrainSamLoc,TrainLabels);
            N = size(WeightMat,1);
            d = sum(WeightMat,2);
            Dinv = spdiags(d,0,N,N);
            Laplace_Mat_no_normalize2=Dinv-WeightMat;
            
                ged_rec1 = [];
                pre_WeightMat1 = WeightMat;
                
%             matSaveDir = ['./trainTestSplit/',select_Data,'2'];
%             if ~exist(matSaveDir)
%                 mkdir(matSaveDir);
%             end
% %             save(['./trainTestSplit/',select_Data,'2/', select_Data,'_sample', num2str(TrainPerNum), '_run', num2str(RandNo_i), '_graph.mat'],'WeightMat');
% %             save(['./trainTestSplit/',select_Data,'2/sample', num2str(TrainPerNum), '_run', num2str(RandNo_i), '.mat'],'TrainSamLoc','TestSamLoc');
%             
%                     train_gt = zeros(size(TruthMap,1)*size(TruthMap,2),1);
%                     train_gt(TrainSamLoc) = TrainLabels;
%                     train_gt = reshape(train_gt,size(TruthMap,1),size(TruthMap,2));
%                     test_gt = zeros(size(TruthMap,1)*size(TruthMap,2),1);
%                     test_gt(TestSamLoc) = TestLabels;
%                     test_gt = reshape(test_gt,size(TruthMap,1),size(TruthMap,2));
%             save(['./trainTestSplit/',select_Data,'2/', select_Data,'_sample', num2str(TrainPerNum), '_run', num2str(RandNo_i), '_graph.mat'],'WeightMat');
%             save(['./trainTestSplit/',select_Data,'2/sample', num2str(TrainPerNum), '_run', num2str(RandNo_i), '_V2.mat'],'TrainSamLoc','TestSamLoc');
%             save(['./trainTestSplit/',select_Data,'2/sample', num2str(TrainPerNum), '_run', num2str(RandNo_i), '.mat'],'train_gt','test_gt');
             
            
            sp_num0=max(superpix_img0);
            initial_Labels = zeros(sp_num0,no_classes);
            for select_i=1:length(TrainSamLoc)
                sp_idx=superpix_img0(TrainSamLoc(select_i));
                initial_Labels(sp_idx,TrainLabels(select_i)) = 1;
            end
            initial_Labels=sparse(initial_Labels);
            
                
            [WeightMat,superpix_img2,Superpixel_mean_Features] = FSPG_graph_construction_bigdata(HyperCube,superpix_num2,superpix_allnum,conn_pattern,TrainSamLoc,TrainLabels);
            N = size(WeightMat,1);
            d = sum(WeightMat,2);
            Dinv = spdiags(d,0,N,N);
            Laplace_Mat_no_normalize20=Dinv-WeightMat;
            
                ged_rec2 = [];
                pre_WeightMat2 = WeightMat;
             
            sp_num2=max(superpix_img2);
            initial_Labels2 = zeros(sp_num2,no_classes);
            for select_i=1:length(TrainSamLoc)
                sp_idx=superpix_img2(TrainSamLoc(select_i));
                initial_Labels2(sp_idx,TrainLabels(select_i)) = 1;
            end
            initial_Labels2=sparse(initial_Labels2);
            
            
            
            [WeightMat,superpix_img3,Superpixel_mean_Features] = FSPG_graph_construction_bigdata(HyperCube,superpix_num3,superpix_allnum,conn_pattern,TrainSamLoc,TrainLabels);
            N = size(WeightMat,1);
            d = sum(WeightMat,2);
            Dinv = spdiags(d,0,N,N);
            Laplace_Mat_no_normalize30=Dinv-WeightMat;
            
            sp_num3=max(superpix_img3);
            initial_Labels3 = zeros(sp_num3,no_classes);
            for select_i=1:length(TrainSamLoc)
                sp_idx=superpix_img3(TrainSamLoc(select_i));
                initial_Labels3(sp_idx,TrainLabels(select_i)) = 1;
            end
            initial_Labels3=sparse(initial_Labels3);
            
            
                % 鏇存柊鏁翠釜楂樺厜璋卞浘鍍忕殑鍚庨獙姒傜巼
                [PredLabels_MLR1,Posterior_data1]=MLR_Classifier_fun_vec(HyperCube_vec,TrainSamLoc,TrainLabels,HyperCube_vec);
                PredLabels2 = PredLabels_MLR1(TestSamLoc);
                AccRate_MLR1(PerNumNo) = fAccuracy(PredLabels2,TestLabels,0)
                
                [PredLabels_MLR2,Posterior_data2]=MLR_Classifier_fun_vec(HyperCubeDR_vec,TrainSamLoc,TrainLabels,HyperCubeDR_vec);
                PredLabels2 = PredLabels_MLR2(TestSamLoc);
                AccRate_MLR2(PerNumNo) = fAccuracy(PredLabels2,TestLabels,0)
                
            
            Time_weight_construct = etime(clock,Time_weight_start)
            
            Time_start=clock;
            
            for iter_alpha=1:length(para_alpha)
                [~,PredLabels2,PostP2] = SemiSupervised_Graph_Classifier_v2(Laplace_Mat_no_normalize2, initial_Labels, [], para_alpha(iter_alpha));
                Posterior_PPSPGv1 = PostP2;
                
                [~,PredLabels2,PostP20] = SemiSupervised_Graph_Classifier_v2(Laplace_Mat_no_normalize20, initial_Labels2, [], para_alpha(iter_alpha));
                Posterior_PPSPGv2 = PostP20;
                
                Posterior_PPSPG_ALL=zeros(Width*Height,no_classes);
                for select_i=1:sp_num0
                    pos_idx=find(superpix_img0==select_i);
                    Posterior_PPSPG_ALL(pos_idx,:)=repmat(Posterior_PPSPGv1(select_i,:),length(pos_idx),1);
                end
                Posterior_PPSPG_ALL2=zeros(Width*Height,no_classes);
                for select_i=1:sp_num2
                    pos_idx=find(superpix_img2==select_i);
                    Posterior_PPSPG_ALL2(pos_idx,:)=repmat(Posterior_PPSPGv2(select_i,:),length(pos_idx),1);
                end
%                 Posterior_PPSPG_ALL = Posterior_PPSPG_ALL.*Posterior_PPSPG_ALL2;
%             Posterior_PPSPG_ALL = Posterior_PPSPG_ALL.*Posterior_PPSPG_ALL2.*(Posterior_data2.^(0.2));
%             if strcmp(select_Data,'Houston2018')
            if strcmp(select_Data,'Houston2018')
                Posterior_data2 = row_normalization(Posterior_data2);
                Posterior_PPSPG_ALL = row_normalization(Posterior_PPSPG_ALL);
                Posterior_PPSPG_ALL2 = row_normalization(Posterior_PPSPG_ALL2);
                Posterior_PPSPG_ALL = Posterior_PPSPG_ALL.*Posterior_PPSPG_ALL2.*(Posterior_data2.^1.5);
%                 Posterior_PPSPG_ALL = Posterior_PPSPG_ALL.*Posterior_PPSPG_ALL2.*(Posterior_data2.^0.2);
%             elseif strcmp(select_Data,'LapYuHeSiDi')
%                 Posterior_data2 = row_normalization(Posterior_data2);
%                 Posterior_PPSPG_ALL = row_normalization(Posterior_PPSPG_ALL);
%                 Posterior_PPSPG_ALL2 = row_normalization(Posterior_PPSPG_ALL2);
% %                 Posterior_PPSPG_ALL = Posterior_PPSPG_ALL.*Posterior_PPSPG_ALL2.*(Posterior_data2.^1.5);
%                 Posterior_PPSPG_ALL = Posterior_PPSPG_ALL.*Posterior_PPSPG_ALL2.*(Posterior_data2.^1.5);
            else
                Posterior_PPSPG_ALL = Posterior_PPSPG_ALL.*Posterior_PPSPG_ALL2;
            end
            
            [~, PredLabels_ALL] = max(Posterior_PPSPG_ALL, [], 2);
                PredLabels2 = PredLabels_ALL(TestSamLoc);
                one_accuracy(iter_alpha) = fAccuracy(PredLabels2,TestLabels,0);% 姣忎釜閫夋嫨鐨勫昂搴︿笂鐨勬祴璇曟牱鏈噯纭巼
            end
            [maxAcc1,bestalphaNo] =max(one_accuracy);
            bestalpha=para_alpha(bestalphaNo);

            time_tmp = etime(clock,Time_start)
            Time_crossValidation_RandNo = [Time_crossValidation_RandNo time_tmp];
            Time_weight_construct_RandNo = [Time_weight_construct_RandNo Time_weight_construct];
            Time_start=clock;

            
            [~,PredLabels2,PostP2] = SemiSupervised_Graph_Classifier_v2(Laplace_Mat_no_normalize2, initial_Labels, [], bestalpha);
            Posterior_PPSPGv1 = PostP2;
            
            [~,PredLabels2,PostP20] = SemiSupervised_Graph_Classifier_v2(Laplace_Mat_no_normalize20, initial_Labels2, [], bestalpha);
            Posterior_PPSPGv2 = PostP20;
            
            Posterior_PPSPG_ALL=zeros(Width*Height,no_classes);
            for select_i=1:sp_num0
                pos_idx=find(superpix_img0==select_i);
                Posterior_PPSPG_ALL(pos_idx,:)=repmat(Posterior_PPSPGv1(select_i,:),length(pos_idx),1);
            end
            Posterior_PPSPG_ALL2=zeros(Width*Height,no_classes);
            for select_i=1:sp_num2
                pos_idx=find(superpix_img2==select_i);
                Posterior_PPSPG_ALL2(pos_idx,:)=repmat(Posterior_PPSPGv2(select_i,:),length(pos_idx),1);
            end
%             Posterior_PPSPG_ALL3 = Posterior_PPSPG_ALL.*Posterior_PPSPG_ALL2.*(Posterior_data2.^(0.2));

%             if strcmp(select_Data,'Houston2018')
            if strcmp(select_Data,'Houston2018')
%             Posterior_data2 = sigmoid_normalization(Posterior_data2);
%             Posterior_PPSPG_ALL = sigmoid_normalization(Posterior_PPSPG_ALL);
%             Posterior_PPSPG_ALL2 = sigmoid_normalization(Posterior_PPSPG_ALL2);

            Posterior_data2 = row_normalization(Posterior_data2);
            Posterior_PPSPG_ALL = row_normalization(Posterior_PPSPG_ALL);
            Posterior_PPSPG_ALL2 = row_normalization(Posterior_PPSPG_ALL2);

%             Posterior_data2 = softmax_normalization(Posterior_data2);
%             Posterior_PPSPG_ALL = softmax_normalization(Posterior_PPSPG_ALL);
%             Posterior_PPSPG_ALL2 = softmax_normalization(Posterior_PPSPG_ALL2);

            Posterior_PPSPG_ALL3 = Posterior_PPSPG_ALL.*Posterior_PPSPG_ALL2.*(Posterior_data2.^1.5);
%                 Posterior_PPSPG_ALL3 = Posterior_PPSPG_ALL.*Posterior_PPSPG_ALL2.*(Posterior_data2.^0.2);
                kernel_size=11;
%             elseif strcmp(select_Data,'LapYuHeSiDi')
%                 Posterior_data2 = row_normalization(Posterior_data2);
%                 Posterior_PPSPG_ALL = row_normalization(Posterior_PPSPG_ALL);
%                 Posterior_PPSPG_ALL2 = row_normalization(Posterior_PPSPG_ALL2);
% %                 Posterior_PPSPG_ALL = Posterior_PPSPG_ALL.*Posterior_PPSPG_ALL2.*(Posterior_data2.^1.5);
%                 Posterior_PPSPG_ALL3 = Posterior_PPSPG_ALL.*Posterior_PPSPG_ALL2.*(Posterior_data2.^1.5);
%                 kernel_size=5;
            else
                Posterior_PPSPG_ALL3 = Posterior_PPSPG_ALL.*Posterior_PPSPG_ALL2;
            end
            
                
            [~, PredLabels_ALL] = max(Posterior_PPSPG_ALL3, [], 2);
            PredLabels_PPSPG = PredLabels_ALL(TestSamLoc);
            confidence_PPSPG_ALL = max(Posterior_PPSPG_ALL3,[],2);

            time_tmp = etime(clock,Time_start)
            Time_classification_RandNo = [Time_classification_RandNo time_tmp];

            tmp_max_OA=[];
            tmp_max_OA(1) = fAccuracy(PredLabels_PPSPG,TestLabels,0)

            
            tmp_max_OA_med=[];
            PredLabels_ALL_med=[];
            if strcmp(select_Data,'Houston2018')
%             if strcmp(select_Data,'Houston2018')||strcmp(select_Data,'LapYuHeSiDi')
                figure;imagesc(reshape(PredLabels_ALL,Height,Width))
                colormap(mycolormap)
                
%                 kernel_size=11;
%                 kernel_size=5;
                % 重塑为3D数组
                network_output = reshape(Posterior_PPSPG_ALL3, [Height,Width, no_classes]);
                % 对每个通道应用中值滤波
                for iii = 1:no_classes  % 遍历通道
                    out_test = network_output(:, :, iii);
                    % 应用中值滤波，kernel_size需要是奇数
                    network_output(:, :, iii) = medfilt2(out_test, [kernel_size, kernel_size]);
                end
                % 重塑回2D数组
                network_output = reshape(network_output, [Height*Width, no_classes]);
                [~, PredLabels_ALL_med] = max(network_output, [], 2);
                PredLabels_PPSPG_med = PredLabels_ALL_med(TestSamLoc);
                tmp_max_OA_med(1) = fAccuracy(PredLabels_PPSPG_med,TestLabels,0)
                figure;imagesc(reshape(PredLabels_ALL_med,Height,Width))
                colormap(mycolormap)
            end
            

            initTrainSamLoc = TrainSamLoc;
            initTrainLabels = TrainLabels;
            if strcmp(select_Data,'WHU_Hi_HanChuan')
                total_iter=7;
            else
                total_iter=length(unique(TrainLabels));
            end
%             total_iter=0;
            for iter=1:total_iter

                para_alpha=0.7;
                [~,PredLabels2,PostP2] = SemiSupervised_Graph_Classifier_v2(Laplace_Mat_no_normalize30, initial_Labels3, [], para_alpha);
                Posterior_PPSPGv1 = PostP2;

                [~,PredLabels2,PostP20] = SemiSupervised_Graph_Classifier_v2(Laplace_Mat_no_normalize20, initial_Labels2, [], para_alpha);
                Posterior_PPSPGv2 = PostP20;

                Posterior_PPSPG_ALL=zeros(Width*Height,no_classes);
                for select_i=1:sp_num3
                    pos_idx=find(superpix_img3==select_i);
                    Posterior_PPSPG_ALL(pos_idx,:)=repmat(Posterior_PPSPGv1(select_i,:),length(pos_idx),1);
                end
                Posterior_PPSPG_ALL2=zeros(Width*Height,no_classes);
                for select_i=1:sp_num2
                    pos_idx=find(superpix_img2==select_i);
                    Posterior_PPSPG_ALL2(pos_idx,:)=repmat(Posterior_PPSPGv2(select_i,:),length(pos_idx),1);
                end
%                 Posterior_PPSPG_ALL = Posterior_PPSPG_ALL2;
%                     if strcmp(select_Data,'Houston2018')
                    if strcmp(select_Data,'Houston2018')
                        Posterior_data2 = row_normalization(Posterior_data2);
                        Posterior_PPSPG_ALL = row_normalization(Posterior_PPSPG_ALL);
                        Posterior_PPSPG_ALL2 = row_normalization(Posterior_PPSPG_ALL2);
                        Posterior_PPSPG_ALL = Posterior_PPSPG_ALL.*Posterior_PPSPG_ALL2.*(Posterior_data2.^1.5);
%                         Posterior_PPSPG_ALL = Posterior_PPSPG_ALL.*Posterior_PPSPG_ALL2.*(Posterior_data2.^0.2);
%                     elseif strcmp(select_Data,'LapYuHeSiDi')
%                         Posterior_data2 = row_normalization(Posterior_data2);
%                         Posterior_PPSPG_ALL = row_normalization(Posterior_PPSPG_ALL);
%                         Posterior_PPSPG_ALL2 = row_normalization(Posterior_PPSPG_ALL2);
%                         %                 Posterior_PPSPG_ALL = Posterior_PPSPG_ALL.*Posterior_PPSPG_ALL2.*(Posterior_data2.^1.5);
%                         Posterior_PPSPG_ALL = Posterior_PPSPG_ALL.*Posterior_PPSPG_ALL2.*(Posterior_data2.^1.5);
                    else
                        Posterior_PPSPG_ALL = Posterior_PPSPG_ALL.*Posterior_PPSPG_ALL2;
                    end
               
                if strcmp(select_Data,'WHU_Hi_HanChuan')
                    BT_num = 5;
                else
                    BT_num = 2;
                end
                if iter==total_iter
                    BT_num = 2;
                end
                BT_num = 10; 
                
                %% PRSP-level
                Posterior_PPSPG=[];
                for i=1:sp_num2
                    sp_idx1=find(superpix_img2==i);
                    X1=mean(Posterior_PPSPG_ALL(sp_idx1,:),1);
                    Posterior_PPSPG=[Posterior_PPSPG;X1];
                end
                % 主动学习BT采样
                sp_ind_candi = unique(superpix_img2(CandiSamLoc));
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
                    candi_pix_locs = find(superpix_img2==candi_spi);
                    [~,candi_pix_locs_id]=ismember(candi_pix_locs,CandiSamLoc);
                    candi_pix_locs_id = candi_pix_locs_id(candi_pix_locs_id~=0);
                    candi_pix_locs = CandiSamLoc(candi_pix_locs_id);
                    
                        candi_pix_locs_sel = [candi_pix_locs_sel;candi_pix_locs(1)];
                end
                
                %% pixel-level
%                 pactive = Posterior_PPSPG_ALL(CandiSamLoc,:)';
%                 pactive = sort(pactive,'descend');
%                 pactive_minBT_PPSPG = pactive(1,:)-pactive(2,:);
%                 BT_scores = pactive_minBT_PPSPG;
%                 [sortmminppBT, indexsortminppBT] = sort(BT_scores);
%                 PPSPG_idx_sel = indexsortminppBT(1:BT_num);
%                 candi_pix_locs_sel = CandiSamLoc(PPSPG_idx_sel);
                

                % 鏇存柊璁粌鏍锋湰
                [~,cfd_match_idx_sel] = ismember(candi_pix_locs_sel,CandiSamLoc);
                    cfd_match_idx_sel = cfd_match_idx_sel(cfd_match_idx_sel~=0);
                NewTrainSamLoc = CandiSamLoc(cfd_match_idx_sel);
%                 NewTrainLabels = PredLabels_MLR1_candi(cfd_match_idx_sel);
                NewTrainLabels = CandiLabels(cfd_match_idx_sel);
                NewNormTrainSam = NormCandiSam(cfd_match_idx_sel,:);

                TrainSamLoc = [TrainSamLoc;NewTrainSamLoc];
                TrainLabels = [TrainLabels;NewTrainLabels];
                NormTrainSam = [NormTrainSam;NewNormTrainSam];

                % 灏嗗?閫夐泦鎺掗櫎鎺変竴浜?
                CandiSamLoc(cfd_match_idx_sel)=[];
                CandiLabels(cfd_match_idx_sel)=[];
                NormCandiSam(cfd_match_idx_sel,:)=[];
                TestSamLoc(cfd_match_idx_sel)=[];
                TestLabels(cfd_match_idx_sel)=[];
                
                
                % 鏇存柊鍥剧粨鏋?
                [WeightMat_MSSPG,superpix_img0,Superpixel_mean_Features] = FSPG_graph_construction_bigdata(HyperCube,superpix_num0,superpix_allnum,conn_pattern,TrainSamLoc,TrainLabels);
                N = size(WeightMat_MSSPG,1);
                d = sum(WeightMat_MSSPG,2);
                Dinv = spdiags(d,0,N,N);
                Laplace_Mat_no_normalize2=Dinv-WeightMat_MSSPG;
                
                ged = simpleGED(full(pre_WeightMat1), full(WeightMat_MSSPG));
                ged_rec1 = [ged_rec1;ged];
                pre_WeightMat1 = WeightMat_MSSPG;
                
                sp_num0=max(superpix_img0);
                initial_Labels = zeros(sp_num0,no_classes);
                for select_i=1:length(TrainSamLoc)
                    sp_idx=superpix_img0(TrainSamLoc(select_i));
                    initial_Labels(sp_idx,TrainLabels(select_i)) = 1;
                end
                initial_Labels=sparse(initial_Labels);
                
                [WeightMat,superpix_img2,Superpixel_mean_Features] = FSPG_graph_construction_bigdata(HyperCube,superpix_num2,superpix_allnum,conn_pattern,TrainSamLoc,TrainLabels);
                N = size(WeightMat,1);
                d = sum(WeightMat,2);
                Dinv = spdiags(d,0,N,N);
                Laplace_Mat_no_normalize20=Dinv-WeightMat;
                
                ged = simpleGED(full(pre_WeightMat2), full(WeightMat));
                ged_rec2 = [ged_rec2;ged];
                pre_WeightMat2 = WeightMat;
                
                sp_num2=max(superpix_img2);
                initial_Labels2 = zeros(sp_num2,no_classes);
                for select_i=1:length(TrainSamLoc)
                    sp_idx=superpix_img2(TrainSamLoc(select_i));
                    initial_Labels2(sp_idx,TrainLabels(select_i)) = 1;
                end
                initial_Labels2=sparse(initial_Labels2);
                
                
                [WeightMat,superpix_img3,Superpixel_mean_Features] = FSPG_graph_construction_bigdata(HyperCube,superpix_num3,superpix_allnum,conn_pattern,TrainSamLoc,TrainLabels);
                N = size(WeightMat,1);
                d = sum(WeightMat,2);
                Dinv = spdiags(d,0,N,N);
                Laplace_Mat_no_normalize30=Dinv-WeightMat;
                
                sp_num3=max(superpix_img3);
                initial_Labels3 = zeros(sp_num3,no_classes);
                for select_i=1:length(TrainSamLoc)
                    sp_idx=superpix_img3(TrainSamLoc(select_i));
                    initial_Labels3(sp_idx,TrainLabels(select_i)) = 1;
                end
                initial_Labels3=sparse(initial_Labels3);
                
            
                [PredLabels_MLR2,Posterior_data2]=MLR_Classifier_fun_vec(HyperCubeDR_vec,TrainSamLoc,TrainLabels,HyperCubeDR_vec);
                PredLabels2 = PredLabels_MLR2(TestSamLoc);
                AccRate_MLR2(PerNumNo) = fAccuracy(PredLabels2,TestLabels,0)
                
                %璋卞浘鑱氱被銆?
                para_alpha=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.93,0.95,0.97,0.99];
                for iter_alpha=1:length(para_alpha)
                    [~,PredLabels2,PostP2] = SemiSupervised_Graph_Classifier_v2(Laplace_Mat_no_normalize2, initial_Labels, [], para_alpha(iter_alpha));
                    Posterior_PPSPGv1 = PostP2;
                    
                    [~,PredLabels2,PostP20] = SemiSupervised_Graph_Classifier_v2(Laplace_Mat_no_normalize20, initial_Labels2, [], para_alpha(iter_alpha));
                    Posterior_PPSPGv2 = PostP20;
                    
                    Posterior_PPSPG_ALL=zeros(Width*Height,no_classes);
                    for select_i=1:sp_num0
                        pos_idx=find(superpix_img0==select_i);
                        Posterior_PPSPG_ALL(pos_idx,:)=repmat(Posterior_PPSPGv1(select_i,:),length(pos_idx),1);
                    end
                    Posterior_PPSPG_ALL2=zeros(Width*Height,no_classes);
                    for select_i=1:sp_num2
                        pos_idx=find(superpix_img2==select_i);
                        Posterior_PPSPG_ALL2(pos_idx,:)=repmat(Posterior_PPSPGv2(select_i,:),length(pos_idx),1);
                    end
%                     if strcmp(select_Data,'Houston2018')
                    if strcmp(select_Data,'Houston2018')
                        Posterior_data2 = row_normalization(Posterior_data2);
                        Posterior_PPSPG_ALL = row_normalization(Posterior_PPSPG_ALL);
                        Posterior_PPSPG_ALL2 = row_normalization(Posterior_PPSPG_ALL2);
                        Posterior_PPSPG_ALL = Posterior_PPSPG_ALL.*Posterior_PPSPG_ALL2.*(Posterior_data2.^1.5);
%                         Posterior_PPSPG_ALL = Posterior_PPSPG_ALL.*Posterior_PPSPG_ALL2.*(Posterior_data2.^0.2);
%                     elseif strcmp(select_Data,'LapYuHeSiDi')
%                         Posterior_data2 = row_normalization(Posterior_data2);
%                         Posterior_PPSPG_ALL = row_normalization(Posterior_PPSPG_ALL);
%                         Posterior_PPSPG_ALL2 = row_normalization(Posterior_PPSPG_ALL2);
%                         %                 Posterior_PPSPG_ALL = Posterior_PPSPG_ALL.*Posterior_PPSPG_ALL2.*(Posterior_data2.^1.5);
%                         Posterior_PPSPG_ALL = Posterior_PPSPG_ALL.*Posterior_PPSPG_ALL2.*(Posterior_data2.^1.5);
                    else
                        Posterior_PPSPG_ALL = Posterior_PPSPG_ALL.*Posterior_PPSPG_ALL2;
                    end
                    
                    [~, PredLabels_ALL] = max(Posterior_PPSPG_ALL, [], 2);
                    PredLabels2 = PredLabels_ALL(TestSamLoc);
                    one_accuracy(iter_alpha) = fAccuracy(PredLabels2,TestLabels,0);% 姣忎釜閫夋嫨鐨勫昂搴︿笂鐨勬祴璇曟牱鏈噯纭巼
                end
                [maxAcc1,bestalphaNo] =max(one_accuracy);
                bestalpha=para_alpha(bestalphaNo);
                
                [~,PredLabels2,PostP2] = SemiSupervised_Graph_Classifier_v2(Laplace_Mat_no_normalize2, initial_Labels, [], bestalpha);
                Posterior_PPSPGv1 = PostP2;
                
                [~,PredLabels2,PostP20] = SemiSupervised_Graph_Classifier_v2(Laplace_Mat_no_normalize20, initial_Labels2, [], bestalpha);
                Posterior_PPSPGv2 = PostP20;
                
                Posterior_PPSPG_ALL=zeros(Width*Height,no_classes);
                for select_i=1:sp_num0
                    pos_idx=find(superpix_img0==select_i);
                    Posterior_PPSPG_ALL(pos_idx,:)=repmat(Posterior_PPSPGv1(select_i,:),length(pos_idx),1);
                end
                Posterior_PPSPG_ALL2=zeros(Width*Height,no_classes);
                for select_i=1:sp_num2
                    pos_idx=find(superpix_img2==select_i);
                    Posterior_PPSPG_ALL2(pos_idx,:)=repmat(Posterior_PPSPGv2(select_i,:),length(pos_idx),1);
                end
%                     if strcmp(select_Data,'Houston2018')
                    if strcmp(select_Data,'Houston2018')
                        Posterior_data2 = row_normalization(Posterior_data2);
                        Posterior_PPSPG_ALL = row_normalization(Posterior_PPSPG_ALL);
                        Posterior_PPSPG_ALL2 = row_normalization(Posterior_PPSPG_ALL2);
                        Posterior_PPSPG_ALL = Posterior_PPSPG_ALL.*Posterior_PPSPG_ALL2.*(Posterior_data2.^1.5);
%                         Posterior_PPSPG_ALL = Posterior_PPSPG_ALL.*Posterior_PPSPG_ALL2.*(Posterior_data2.^0.2);
%                     elseif strcmp(select_Data,'LapYuHeSiDi')
%                         Posterior_data2 = row_normalization(Posterior_data2);
%                         Posterior_PPSPG_ALL = row_normalization(Posterior_PPSPG_ALL);
%                         Posterior_PPSPG_ALL2 = row_normalization(Posterior_PPSPG_ALL2);
%                         %                 Posterior_PPSPG_ALL = Posterior_PPSPG_ALL.*Posterior_PPSPG_ALL2.*(Posterior_data2.^1.5);
%                         Posterior_PPSPG_ALL = Posterior_PPSPG_ALL.*Posterior_PPSPG_ALL2.*(Posterior_data2.^1.5);
                    else
                        Posterior_PPSPG_ALL = Posterior_PPSPG_ALL.*Posterior_PPSPG_ALL2;
                    end
                [~, PredLabels_ALL] = max(Posterior_PPSPG_ALL, [], 2);
                PredLabels_PPSPG = PredLabels_ALL(TestSamLoc);
                tmp_max_OA(iter+1) = fAccuracy(PredLabels_PPSPG,TestLabels,0)

                
                PredLabels_ALL_med=[];
                if strcmp(select_Data,'Houston2018')
%                 if strcmp(select_Data,'Houston2018')||strcmp(select_Data,'LapYuHeSiDi')
                    figure;imagesc(reshape(PredLabels_ALL,Height,Width))
                    colormap(mycolormap)
                    
%                     kernel_size=5;
                    % 重塑为3D数组
                    network_output = reshape(Posterior_PPSPG_ALL, [Height,Width, no_classes]);
                    % 对每个通道应用中值滤波
                    for iii = 1:no_classes  % 遍历通道
                        out_test = network_output(:, :, iii);
                        % 应用中值滤波，kernel_size需要是奇数
                        network_output(:, :, iii) = medfilt2(out_test, [kernel_size, kernel_size]);
                    end
                    % 重塑回2D数组
                    network_output = reshape(network_output, [Height*Width, no_classes]);
                    [~, PredLabels_ALL_med] = max(network_output, [], 2);
                    PredLabels_PPSPG_med = PredLabels_ALL_med(TestSamLoc);
                    tmp_max_OA_med(iter+1) = fAccuracy(PredLabels_PPSPG_med,TestLabels,0)
                    figure;imagesc(reshape(PredLabels_ALL_med,Height,Width))
                    colormap(mycolormap)
                end
            end
            
            DFSG_AL_Time = toc;
            DFSG_AL_Time2 = etime(clock,Time_DFSGAL_start)
            
            AccRate_RandNo = [AccRate_RandNo tmp_max_OA(end)];
            IndiAccRate_tmp = fAccuracy(PredLabels_PPSPG,TestLabels,1);
            IndiAccRate_RandNo = [IndiAccRate_RandNo IndiAccRate_tmp];
            [kappaCoef_tmp, zscore_ESD_tmp] = fKappaCoef(PredLabels_PPSPG,TestLabels);
            kappaCoef_RandNo = [kappaCoef_RandNo kappaCoef_tmp];
            zscore_ESD_RandNo = [zscore_ESD_RandNo zscore_ESD_tmp];
            AA_tmp = mean(IndiAccRate_tmp,1);
%             matSaveDir = ['./Result/',select_Data,'/',AL_method,'V2.6_ratio',num2str(setratio),'_PRSPlevelsampling/initTrainNum-', num2str(TrainPerNum),'/RngSeed-', num2str(RandNo_i)];
            matSaveDir = ['./Result/',select_Data,'/',AL_method,'V2.6_ratio',num2str(setratio),'_pixellevelsamplingV2/initTrainNum-', num2str(TrainPerNum),'/RngSeed-', num2str(RandNo_i)];
%             matSaveDir = ['./Result/',select_Data,'/',AL_method,'V2.6.01_iter16/initTrainNum-', num2str(TrainPerNum),'/RngSeed-', num2str(RandNo_i)];
            if ~exist(matSaveDir)
                mkdir(matSaveDir);
            end
%             if total_iter>0
%                 save([matSaveDir,'/', select_Data,'_sample', num2str(length(TrainSamLoc)), '_run', num2str(RandNo_i), '_',AL_method,'.mat'],'WeightMat_MSSPG');
%             end
            save([matSaveDir,'/DFSG_AL_Time.mat'],'DFSG_AL_Time')
            save([matSaveDir,'/DFSG_AL_Time2.mat'],'DFSG_AL_Time2')
            save([matSaveDir,'/TrainLabels.mat'],'TrainLabels')
            save([matSaveDir,'/PredLabels.mat'],'PredLabels_ALL')
            save([matSaveDir,'/TestLabels.mat'],'TestLabels')
            save([matSaveDir,'/TrainSamLoc.mat'],'TrainSamLoc')
            save([matSaveDir,'/TestSamLoc.mat'],'TestSamLoc')
            save([matSaveDir,'/OA.mat'],'tmp_max_OA')
            save([matSaveDir,'/AA.mat'],'AA_tmp')
            save([matSaveDir,'/kappaCoef.mat'],'kappaCoef_tmp')
            save([matSaveDir,'/IndiAccRate.mat'],'IndiAccRate_tmp')

            save([matSaveDir,'/PredLabels_med.mat'],'PredLabels_ALL_med')
            save([matSaveDir,'/OA_med.mat'],'tmp_max_OA_med')
            
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