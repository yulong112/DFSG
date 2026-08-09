
clc
close all
clear all
% matlabpath(pathdef)
addpath(genpath('./MLR_toolbox'))
addpath(genpath('./Graph_toolbox'))
addpath(genpath('./Utils_DFSG'));
addpath(genpath('./SuperPixel_toolbox'));

%% input param
sg_num=1;
mini_sp_num=15;
interval_sp_num=300;
% sp_num_vec=100:100:1500;
sp_num_vec=2400:200:3200;
conn_pattern='combine';

kernel_h=15;

select_Data='Zaoyuan';
select_Data='IndianaPine';

        TrainSamPerOrNum = [3];
            AL_method = 'MSSPG-singleAL';

if strcmp(select_Data,'IndianaPine')
    superpix_allnum_vec=[[300,600];];
    %% IndianaPine 鏁版嵁colormap
    mycolormap = [  0    0    0.5;           %鑳屾櫙钃?
        %     1    0.75  0.75;                     %
        1    0.75    0;                      %娣卞湡榛?
        1    0    0;                         %绾?
        0    1    0;                         %缁?
        1    0    1;                         %娴呯传绮夌孩鑹?
        0  0.5    1;                         %涓嶆繁涓嶆祬钃濊壊
        %     0.75    0    0.8                        %
        0    1    1;                         %娣℃祬钃?
        %     0  0.75  0.5;
        1    1    0.5;                       %娴呯背榛?
        0.5 0.25    0;                       %妫?
        0.5    0  0.5;                       %娣辩传鑹?
        0   0   0.8;                         %娣辫摑鑹?
        0.8   0.8   1;                       %鐏拌壊
        0.8   0.8   0;]                       %妫曢粍鑹?
    %% IndianaPine 鏁版嵁
    load IndianaPine
    load IndianaPine_LabelMap
    HyperCube= IndianaPine;
    TruthMap = IndianaPine_LabelMap;
    clear IndianaPine IndianaPine_LabelMap
    YStart = 1; % left-up point
    XStart = 1;
    YEnd = 145; % right-down point
    XEnd = 145;
    %DelBand = [1:5, 95, 96, 101:111,148:166,216:220];
    DelBand = [1:5, 101:111,148:166,216:220];
    DelClass = [1]; % 绗竴绫绘?鏄儗鏅紝绫绘爣涓?
%     DelClass = [1 2 8 10 17]; % 绗竴绫绘?鏄儗鏅紝绫绘爣涓?
    TrainPercent = 0.01;
    PreliminarySteps1
    
    [Height Width Bands] = size(HyperCube);
    
%% add noise 
%     OriData3 = HyperCube/255;
%     nSig = 3/255;
%     nSig = 0.15;
%     % nSig = 255/255;
%     noiselevel = nSig*ones(1,size(OriData3,3));
%     % %----------------------------------------------noise simulated---------------------------------------------------------
%     oriData3_noise = OriData3;
%     % % Gaussian noise
% %     for i =1:Bands
% %         randn('seed',i);
% %         oriData3_noise(:,:,i)=OriData3(:,:,i)  + noiselevel(i)*randn(Height, Width);
% %         %        mixed(vv,:)=addnoise(mixed(vv,:),'mg',ratio);
% %     end
%     % % salt & pepper noise
%     nSig = 0.01;
%     oriData3_noise=imnoise(OriData3,'salt & pepper', 0.09);
%     HyperCube = 255*oriData3_noise;
%     
%     figure;imshow(mat2gray(HyperCube(:,:,[41,1,153])));
    
sigma_l=90;
weight_beta=0.9;
sp_num_vec=[2000];
superpix_allnum=[300,600];
% sp_num_vec=800:200:1400;
    superpix_num0=floor(Height*Width/4);

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
%     superpix_num0=80*200;
    
elseif strcmp(select_Data,'Zaoyuan')
    superpix_allnum_vec=[[900,1500];];
    superpix_allnum=[900,1500];
    %% Zaoyuan 鏁版嵁colormap
    mycolormap = [  0    0    0.5;           %鑳屾櫙钃?
        1    0.75    0;                      %娣卞湡榛?
        1    0    0;                         %绾?
        0    1    0;                         %缁?
        1    0    1;                         %娴呯传绮夌孩鑹?
        0.8   0.8   1;                       %鐏拌壊
        0    1    1;                         %娣℃祬钃?
        1    1    0.5;                       %娴呯背榛?
        0.5 0.25    0;]                      %妫?
    %% Zaoyuan 鏁版嵁
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
    DelClass = [1]; % 绗竴绫绘?鏄儗鏅紝绫绘爣涓?
    TrainPercent = 0.01;
    PreliminarySteps1
    
%% add noise    
%     OriData3 = HyperCube/255;
%     nSig = 3/255;
%     nSig = 0.01;
%     % nSig = 255/255;
%     noiselevel = nSig*ones(1,size(OriData3,3));
%     % %----------------------------------------------noise simulated---------------------------------------------------------
%     oriData3_noise = OriData3;
%     % % Gaussian noise
% %     for i =1:Bands
% %         randn('seed',i);
% %         oriData3_noise(:,:,i)=OriData3(:,:,i)  + noiselevel(i)*randn(Height, Width);
% %         %        mixed(vv,:)=addnoise(mixed(vv,:),'mg',ratio);
% %     end
%     % % salt & pepper noise
% %     nSig = 0.01;
% %     oriData3_noise=imnoise(OriData3,'salt & pepper', 0.01);
%     
%     HyperCube = 255*oriData3_noise;
% %     HyperCube=imnoise(mat2gray(HyperCube),'gaussian',0,0.01);
% %     HyperCube=mat2gray(HyperCube);
% %     HyperCube=imnoise(mat2gray(HyperCube),'salt & pepper', 0.05);
% %     HyperCube=fmapminmax(HyperCube);
%     
%     figure;imshow(mat2gray(HyperCube(:,:,[31,1,71])));
    
sigma_l=100;
weight_beta=0.8;
sp_num_vec=[2400];
    superpix_num0=floor(Height*Width/4);
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
    
    
    SP_Gabor_data=[];
    SP_Gabor_data2=[];
    SP_Gabor_data3=[];
    SP_Gabor_data4=[];
%     HyperCube2 = Gaussian_GaussianDerivative(HyperCube);
    
%     [Height,Width,Bands] = size(HyperCube2);
%     HyperCube2_vec = reshape(HyperCube2, Width*Height, Bands); %3D鍙?2D
%     HyperCube2DR_vec = pca_row(HyperCube2_vec,30);
%     [Height,Width,Bands] = size(HyperCube);
%     HyperCube_vec = reshape(HyperCube, Width*Height, Bands); %3D鍙?2D
%     HyperCubeDR_vec = pca_row(HyperCube_vec,30);
    
    
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
        
        if strcmp(select_Data,'Zaoyuan')
            RandNo = [4,5,8,13,20];
            RandNo = 1:5;
        else
            RandNo = 1:5;
        end
        for RandNo_i=RandNo
            [NormTrainSam, NormTestSam, TrainLabels, TestLabels, TrainSamLoc, TestSamLoc] = fSetTrainTestSam_Neighbor_V2...
                (HyperCube, TruthMap, SelClassNo, NeighborMask, RandNo_i, NumClassSam);
            
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
% %             load([matSaveDir,'/TrainLabels.mat'],'TrainLabels_save')
% %             load([matSaveDir,'/TestLabels.mat'],'TestLabels_save')
% %             load([matSaveDir,'/TrainSamLoc.mat'],'TrainSamLoc_save')
% %             load([matSaveDir,'/TestSamLoc.mat'],'TestSamLoc_save')
% % %             TrainLabels=TrainLabels_save{end};
% % %             TestLabels=TestLabels_save{end};
% %             TrainSamLoc=TrainSamLoc_save{end};
% % %             TestSamLoc=TestSamLoc_save{end};
% %             AvailableLoc = find(TruthMap1D>0);
% %             TestSamLoc = setdiff(AvailableLoc,TrainSamLoc);
% %             TrainLabels = TruthMap1D(TrainSamLoc);
% %             TestLabels = TruthMap1D(TestSamLoc);
            
            InitTrainSamLoc=TrainSamLoc;
            InitTrainLabels=TrainLabels;
            AvailableLoc=[TrainSamLoc;TestSamLoc];
            
            % 璁惧畾鍊欓?闆嗭紝鍙渶瑕佹墍鏈夌殑娴嬭瘯鏍锋湰鍜屾祴璇曟牱鏈綅缃?
            CandiSamLoc=TestSamLoc;
            CandiLabels=TestLabels;

            tic;
            Time_DFSGAL_start=clock;
            %% 瓒呭儚绱犺氨鍥捐仛绫汇?
%             Time_weight_construct=[];
            one_accuracy=[];
            para_alpha=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.93,0.95,0.97,0.99];
            Time_weight_start=clock;
            
            [WeightMat,superpix_img0,Superpixel_mean_Features] = FSPG_graph_construction_bigdata(HyperCube,superpix_num0,superpix_allnum,conn_pattern,TrainSamLoc,TrainLabels);
            N = size(WeightMat,1);
            d = sum(WeightMat,2);
            Dinv = spdiags(d,0,N,N);
            Laplace_Mat_no_normalize2=Dinv-WeightMat;
            
%             matSaveDir = ['./trainTestSplit/',select_Data,'2'];
%             if ~exist(matSaveDir)
%                 mkdir(matSaveDir);
%             end
%             save(['./trainTestSplit/',select_Data,'2/', select_Data,'_sample', num2str(TrainPerNum), '_run', num2str(RandNo_i), '_graph.mat'],'WeightMat');
%             save(['./trainTestSplit/',select_Data,'2/sample', num2str(TrainPerNum), '_run', num2str(RandNo_i), '.mat'],'TrainSamLoc','TestSamLoc');
            
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
            
                
            WeightMat = FSPG_graph_construction(HyperCube,superpix_allnum,conn_pattern,TrainSamLoc,TrainLabels);
            N = size(WeightMat,1);
            d = sum(WeightMat,2);
            Dinv = spdiags(d,0,N,N);
            Laplace_Mat_no_normalize20=Dinv-WeightMat;
            
            initial_Labels2 = zeros(Height*Width,no_classes);
            for select_i=1:length(TrainSamLoc)
                initial_Labels2(TrainSamLoc(select_i),TrainLabels(select_i)) = 1;
            end
            initial_Labels2=sparse(initial_Labels2);
            
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
                Posterior_PPSPG_ALL2=Posterior_PPSPGv2;
                Posterior_PPSPG_ALL = Posterior_PPSPG_ALL.*Posterior_PPSPG_ALL2;
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
            Posterior_PPSPG_ALL2=Posterior_PPSPGv2;
            Posterior_PPSPG_ALL = Posterior_PPSPG_ALL.*Posterior_PPSPG_ALL2;
            
                
            [~, PredLabels_ALL] = max(Posterior_PPSPG_ALL, [], 2);
            PredLabels_PPSPG = PredLabels_ALL(TestSamLoc);
            confidence_PPSPG_ALL = max(Posterior_PPSPG_ALL,[],2);

            time_tmp = etime(clock,Time_start)
            Time_classification_RandNo = [Time_classification_RandNo time_tmp];

            tmp_max_OA=[];
            tmp_max_OA(1) = fAccuracy(PredLabels_PPSPG,TestLabels,0)

            
            initTrainSamLoc = TrainSamLoc;
            initTrainLabels = TrainLabels;
            if strcmp(select_Data,'WHU_Hi_HanChuan')|| strcmp(select_Data,'IndianaPine')
                total_iter=7;
            else
                total_iter=length(unique(TrainLabels));
            end
%             total_iter=0;
            for iter=1:total_iter
%                     [PredLabels,posterior]=MLR_Classifier_fun(HyperCube,TrainSamLoc,TrainLabels,NormCandiSam);
                %                     AccRate_MLR_prev = fAccuracy(PredLabels,TestLabels,0)

                para_alpha=0.99;
%                 para_alpha=0.7;
                alpha_param=para_alpha;
%                 [~,PredLabels2,PostP2] = SemiSupervised_Graph_Classifier_v2(Laplace_Mat_no_normalize20, initial_Labels2, [], para_alpha);
% %                 [~,PredLabels2,PostP2] = SemiSupervised_Graph_Classifier_v2(Laplace_Mat_no_normalize2, initial_Labels, [], para_alpha);
% %                 if strcmp(AL_method, 'MSSPG-singleAL')
%                     Posterior_PPSPG = PostP2;
%                 elseif strcmp(AL_method, 'MSSPG-singleAL-multiPost')
%                     [~,PredLabels2,PostP1] = SemiSupervised_Graph_Classifier_v2(Laplace_Mat_no_normalize, initial_Labels, [], para_alpha);
%                     [~,PredLabels2,PostP3] = SemiSupervised_Graph_Classifier_v2(Laplace_Mat_no_normalize3, initial_Labels, [], para_alpha);
%                     Posterior_PPSPG = PostP1.*PostP2.*PostP3;
%                 end
                [~,PredLabels2,PostP2] = SemiSupervised_Graph_Classifier_v2(Laplace_Mat_no_normalize2, initial_Labels, [], para_alpha);
                Posterior_PPSPGv1 = PostP2;

                [~,PredLabels2,PostP20] = SemiSupervised_Graph_Classifier_v2(Laplace_Mat_no_normalize20, initial_Labels2, [], para_alpha);
                Posterior_PPSPGv2 = PostP20;

                Posterior_PPSPG=zeros(Width*Height,no_classes);
                for select_i=1:sp_num0
                    pos_idx=find(superpix_img0==select_i);
                    Posterior_PPSPG(pos_idx,:)=repmat(Posterior_PPSPGv1(select_i,:),length(pos_idx),1);
                end
                Posterior_PPSPG = Posterior_PPSPG.*Posterior_PPSPGv2;



                if strcmp(select_Data,'WHU_Hi_HanChuan')|| strcmp(select_Data,'IndianaPine')
                    BT_num = 10;
                else
                    BT_num = 2;
                end
                if iter==total_iter
                    BT_num = 2;
                end
                 BT_num = 2;
                % 主动学习BT采样
                pactive = Posterior_PPSPG(CandiSamLoc,:)';
                pactive = sort(pactive,'descend');
                pactive_minBT_PPSPG = pactive(1,:)-pactive(2,:);
                BT_scores = pactive_minBT_PPSPG;

                [sortmminppBT, indexsortminppBT] = sort(BT_scores);
                PPSPG_idx_sel = indexsortminppBT(1:BT_num);

                NewTrainSamLoc = CandiSamLoc(PPSPG_idx_sel);
%                 NewTrainLabels = PredLabels_MLR1_candi(cfd_match_idx_sel);
                NewTrainLabels = CandiLabels(PPSPG_idx_sel);

                TrainSamLoc = [TrainSamLoc;NewTrainSamLoc];
                TrainLabels = [TrainLabels;NewTrainLabels];

                % 灏嗗?閫夐泦鎺掗櫎鎺変竴浜?
                CandiSamLoc(PPSPG_idx_sel)=[];
                CandiLabels(PPSPG_idx_sel)=[];
                TestSamLoc(PPSPG_idx_sel)=[];
                TestLabels(PPSPG_idx_sel)=[];
                
                
                % 鏇存柊鍥剧粨鏋?
                [WeightMat_MSSPG,superpix_img0,Superpixel_mean_Features] = FSPG_graph_construction_bigdata(HyperCube,superpix_num0,superpix_allnum,conn_pattern,TrainSamLoc,TrainLabels);
                N = size(WeightMat_MSSPG,1);
                d = sum(WeightMat_MSSPG,2);
                Dinv = spdiags(d,0,N,N);
                Laplace_Mat_no_normalize2=Dinv-WeightMat_MSSPG;
                
                
                sp_num0=max(superpix_img0);
                initial_Labels = zeros(sp_num0,no_classes);
                for select_i=1:length(TrainSamLoc)
                    sp_idx=superpix_img0(TrainSamLoc(select_i));
                    initial_Labels(sp_idx,TrainLabels(select_i)) = 1;
                end
                initial_Labels=sparse(initial_Labels);
                
                
                WeightMat = FSPG_graph_construction(HyperCube,superpix_allnum,conn_pattern,TrainSamLoc,TrainLabels);
                N = size(WeightMat,1);
                d = sum(WeightMat,2);
                Dinv = spdiags(d,0,N,N);
                Laplace_Mat_no_normalize20=Dinv-WeightMat;
                
                initial_Labels2 = zeros(Height*Width,no_classes);
                for select_i=1:length(TrainSamLoc)
                    initial_Labels2(TrainSamLoc(select_i),TrainLabels(select_i)) = 1;
                end
                initial_Labels2=sparse(initial_Labels2);
                
                
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
                    Posterior_PPSPG_ALL2=Posterior_PPSPGv2;
                    Posterior_PPSPG_ALL = Posterior_PPSPG_ALL.*Posterior_PPSPG_ALL2;
                    
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
                Posterior_PPSPG_ALL2=Posterior_PPSPGv2;
                Posterior_PPSPG_ALL = Posterior_PPSPG_ALL.*Posterior_PPSPG_ALL2;
                [~, PredLabels_ALL] = max(Posterior_PPSPG_ALL, [], 2);
                PredLabels_PPSPG = PredLabels_ALL(TestSamLoc);
                tmp_max_OA(iter+1) = fAccuracy(PredLabels_PPSPG,TestLabels,0)

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
%             matSaveDir = ['./Result/',select_Data,'/',AL_method,'V2.6.01alpha',num2str(alpha_param),'/initTrainNum-', num2str(TrainPerNum),'/RngSeed-', num2str(RandNo_i)];
%             matSaveDir = ['./Result/',select_Data,'/',AL_method,'V2.6_iter16/initTrainNum-', num2str(TrainPerNum),'/RngSeed-', num2str(RandNo_i)];
            matSaveDir = ['./Result/',select_Data,'/',AL_method,'GF_discuss_Gnoise0/initTrainNum-', num2str(TrainPerNum),'/RngSeed-', num2str(RandNo_i)];
            if ~exist(matSaveDir)
                mkdir(matSaveDir);
            end
            if total_iter>0
%                 save([matSaveDir,'/', select_Data,'_sample', num2str(length(TrainSamLoc)), '_run', num2str(RandNo_i), '_',AL_method,'.mat'],'WeightMat_MSSPG');
            end
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