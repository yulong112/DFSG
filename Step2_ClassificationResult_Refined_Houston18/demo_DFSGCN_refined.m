%% v1,鍙洿鏂板師濮嬬壒寰丮LR鐨勮缁冩牱鏈紝
%% 鐜拌薄锛氳繕鏄細鐣ュ井涓嬮檷銆傝鏄庯細閲囨牱瀵瑰師濮嬬壒寰丮LR鐨勫奖鍝嶈緝灏忥紝浣嗚繕鏄渶瑕佽皟鏁?
%% 鐒跺悗鏀规垚绌蜂妇锛屾瘡杞噰鏍?6涓偣锛屼娇鐩爣鍑芥暟鏈?皬锛岀洰鏍囧嚱鏁版槸璁粌闆嗕笂鐨勪氦鍙夌喌

clc
close all
clear all
addpath(genpath('../DFSG-AL_MATLAB'))

%% input param
% select_Data='zhengzhou_20191110';
% select_Data='WHU_Hi_HanChuan';
% select_Data='LapYuHeSiDi';
select_Data='Houston2018';

%         TrainSamPerOrNum = [3,9];
%         TrainSamPerOrNum = [3,5,7,10,15,20,50,100];
%         TrainSamPerOrNum = [3,5,7,10];
%         TrainSamPerOrNum = [300];
%         TrainSamPerOrNum = [100];
%         TrainSamPerOrNum = [10,15,20,50];
        TrainSamPerOrNum = [150];
        TrainSamPerOrNum = [20];

if strcmp(select_Data,'Houston2018')
    addpath(genpath('../DFSG-AL_MATLAB/Houston2018'))
    
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
    rmpath(genpath('../DFSG-AL_MATLAB/Houston2018'));
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
    
    
    SP_Gabor_data=[];
    SP_Gabor_data2=[];
    SP_Gabor_data3=[];
    SP_Gabor_data4=[];
%     HyperCube2 = Gaussian_GaussianDerivative(HyperCube);
    
%     [Height,Width,Bands] = size(HyperCube2);
%     HyperCube2_vec = reshape(HyperCube2, Width*Height, Bands); %3D鍙?2D
%     HyperCube2DR_vec = pca_row(HyperCube2_vec,30);
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
        
        AccRate_RandNo_med = [];
        IndiAccRate_RandNo_med = [];
        kappaCoef_RandNo_med = [];
        
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
            
            matSaveDir = ['../DFSG-AL_MATLAB/trainTestSplit/',select_Data,'2'];
            if ~exist(matSaveDir)
                mkdir(matSaveDir);
            end
%             save(['./trainTestSplit/',select_Data,'2/', select_Data,'_sample', num2str(TrainPerNum), '_run', num2str(RandNo_i), '_graph.mat'],'WeightMat');
%             save(['./trainTestSplit/',select_Data,'2/sample', num2str(TrainPerNum), '_run', num2str(RandNo_i), '.mat'],'TrainSamLoc','TestSamLoc');
            
                    train_gt = zeros(size(TruthMap,1)*size(TruthMap,2),1);
                    train_gt(TrainSamLoc) = TrainLabels;
                    train_gt = reshape(train_gt,size(TruthMap,1),size(TruthMap,2));
                    test_gt = zeros(size(TruthMap,1)*size(TruthMap,2),1);
                    test_gt(TestSamLoc) = TestLabels;
                    test_gt = reshape(test_gt,size(TruthMap,1),size(TruthMap,2));
%             save(['./trainTestSplit/',select_Data,'2/', select_Data,'_sample', num2str(TrainPerNum), '_run', num2str(RandNo_i), '_graph.mat'],'WeightMat');
%             save(['./trainTestSplit/',select_Data,'2/sample', num2str(TrainPerNum), '_run', num2str(RandNo_i), '_V2.mat'],'TrainSamLoc','TestSamLoc');
%             save(['./trainTestSplit/',select_Data,'2/sample', num2str(TrainPerNum), '_run', num2str(RandNo_i), '.mat'],'train_gt','test_gt');
             
%             load(['./',select_Data,'_/num', num2str(TrainPerNum), 'LeakyReLU_251025/iter', num2str(RandNo_i), '/softmax.mat'])
            load(['./',select_Data,'_/num', num2str(TrainPerNum), 'LeakyReLU_251025/iter', num2str(RandNo_i), '/fused_fea.mat'])
%                 [Height,Width,channels] = size(softmax);
%                 Posterior_PPSPG_ALL = reshape(double(softmax), Height*Width,channels);
                [Height,Width,channels] = size(fused_fea);
                Posterior_PPSPG_ALL2 = reshape(double(fused_fea), Height*Width,channels);
            
                % 鏇存柊鏁翠釜楂樺厜璋卞浘鍍忕殑鍚庨獙姒傜巼
                [PredLabels_MLR1,Posterior_data1]=MLR_Classifier_fun_vec(HyperCube_vec,TrainSamLoc,TrainLabels,HyperCube_vec);
                PredLabels2 = PredLabels_MLR1(TestSamLoc);
                AccRate_MLR1(PerNumNo) = fAccuracy(PredLabels2,TestLabels,0)
                
                [PredLabels_MLR2,Posterior_data2]=MLR_Classifier_fun_vec(HyperCubeDR_vec,TrainSamLoc,TrainLabels,HyperCubeDR_vec);
                PredLabels2 = PredLabels_MLR2(TestSamLoc);
                AccRate_MLR2(PerNumNo) = fAccuracy(PredLabels2,TestLabels,0)
                
            Posterior_data2 = row_normalization(Posterior_data2);
            Posterior_PPSPG_ALL2 = row_normalization(Posterior_PPSPG_ALL2);

            Posterior_PPSPG_ALL3 = (Posterior_PPSPG_ALL2).*(Posterior_data2.^0.1);
            [~, PredLabels_ALL] = max(Posterior_PPSPG_ALL3, [], 2);
            PredLabels_PPSPG = PredLabels_ALL(TestSamLoc);

            
            tmp_max_OA=[];
            tmp_max_OA(1) = fAccuracy(PredLabels_PPSPG,TestLabels,0)

            tmp_max_OA_med=[];
            PredLabels_ALL_med=[];
            if strcmp(select_Data,'Houston2018')
                figure;imagesc(reshape(PredLabels_ALL,Height,Width))
                colormap(mycolormap)
                
                kernel_size=11;
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
            
            AccRate_RandNo = [AccRate_RandNo tmp_max_OA(end)];
            IndiAccRate_tmp = fAccuracy(PredLabels_PPSPG,TestLabels,1);
            IndiAccRate_RandNo = [IndiAccRate_RandNo IndiAccRate_tmp];
            [kappaCoef_tmp, zscore_ESD_tmp] = fKappaCoef(PredLabels_PPSPG,TestLabels);
            kappaCoef_RandNo = [kappaCoef_RandNo kappaCoef_tmp];
            
            AccRate_RandNo_med = [AccRate_RandNo_med tmp_max_OA_med(end)];
            IndiAccRate_tmp_med = fAccuracy(PredLabels_PPSPG_med,TestLabels,1);
            IndiAccRate_RandNo_med = [IndiAccRate_RandNo_med IndiAccRate_tmp_med];
            [kappaCoef_tmp, zscore_ESD_tmp] = fKappaCoef(PredLabels_PPSPG_med,TestLabels);
            kappaCoef_RandNo_med = [kappaCoef_RandNo_med kappaCoef_tmp];
            zscore_ESD_RandNo = [zscore_ESD_RandNo zscore_ESD_tmp];
            AA_tmp = mean(IndiAccRate_tmp_med,1);
            matSaveDir = ['./',select_Data,'/DFSGCN_pixellevel/initTrainNum-', num2str(TrainPerNum),'/RngSeed-', num2str(RandNo_i)];
            if ~exist(matSaveDir)
                mkdir(matSaveDir);
            end
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
        
        AccRate_med(:,PerNumNo) = mean(AccRate_RandNo_med,2)
        IndiAccRate_med(:,PerNumNo) = mean(IndiAccRate_RandNo_med,2)
        kappaCoef_med(:,PerNumNo) = mean(kappaCoef_RandNo_med,2)
        
        Time_crossValidation(:,PerNumNo) = mean(Time_crossValidation_RandNo,2)
        Time_classification(:,PerNumNo) = mean(Time_classification_RandNo,2)
        
        Time_computeWeight(:,PerNumNo) = mean(Time_weight_construct_RandNo,2);
                        
    end