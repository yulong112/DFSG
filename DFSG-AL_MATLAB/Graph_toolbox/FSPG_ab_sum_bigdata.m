function [S_W_no_normalize,superpix_img0] = FSPG_ab_sum_bigdata(HyperCube,superpix_num0,superpix_allnum,conn_pattern,TrainSamLoc,TrainLabels)
[Height,Width,Bands] = size(HyperCube);
HyperCube_vec = reshape(HyperCube, Width*Height, Bands); %3Då?2D
S_W=[];
WeightMatfuse={};
fuse_idx=0;

%% ³¬ÏñËØ0¡£
superpix_img0 = superpixel_cut(HyperCube,superpix_num0);
% superpix_img0 = superpixel_cut_pca(HyperCube,superpix_num0);
sp_num = max(superpix_img0);
for select_i=1:sp_num
    pos_idx=find(superpix_img0==select_i);
    spTrainflag=ismember(TrainSamLoc,pos_idx);
    spTrainSamLoc = TrainSamLoc(spTrainflag);
    spTrainLabels = TrainLabels(spTrainflag);
    if length(spTrainSamLoc)
        spTrainLabels=yl_label_reranging(spTrainLabels)+1;
        if max(spTrainLabels)>1
            % è¶…è¿‡ä¸¤ç±»ï¼Œåˆ†å‰²è¶…åƒç´ ä¸ºå¤šå?
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
            % å°†å…¶ä¸­çš„æŸäº›è¶…åƒç´ åºå·å¢žåŠ?
            for splabel_i=2:max(PredLabels_sp)
                sploc_i = find(PredLabels_sp==splabel_i);
                max_sp_ind = max(superpix_img0);
                superpix_img0(pos_idx(sploc_i))=max_sp_ind+1;
            end
        end
    end
end
superpix_img0=yl_label_reranging(superpix_img0)+1;

sp_num0=max(superpix_img0);
[Height,Width,Bands] = size(HyperCube);
ALL_data = reshape(HyperCube, Width*Height, Bands); %3D±ä 2D

Superpixel_mean_Features=[];
for i=1:sp_num0
    sp_idx1=find(superpix_img0==i);
    X1=mean(ALL_data(sp_idx1,:),1);
    Superpixel_mean_Features=[Superpixel_mean_Features;X1];
end

for superpix_num=superpix_allnum
    %% ³¬ÏñËØÆ×Í¼¾ÛÀà¡£
    superpix_img = superpixel_cut(HyperCube,superpix_num);
%     superpix_img = superpixel_cut_pca(HyperCube,superpix_num);
    sp_num = max(superpix_img);
    for select_i=1:sp_num
        pos_idx=find(superpix_img==select_i);
        spTrainflag=ismember(TrainSamLoc,pos_idx);
        spTrainSamLoc = TrainSamLoc(spTrainflag);
        spTrainLabels = TrainLabels(spTrainflag);
        if length(spTrainSamLoc)
            spTrainLabels=yl_label_reranging(spTrainLabels)+1;
            if max(spTrainLabels)>1
                % è¶…è¿‡ä¸¤ç±»ï¼Œåˆ†å‰²è¶…åƒç´ ä¸ºå¤šå?
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
                % å°†å…¶ä¸­çš„æŸäº›è¶…åƒç´ åºå·å¢žåŠ?
                for splabel_i=2:max(PredLabels_sp)
                    sploc_i = find(PredLabels_sp==splabel_i);
                    max_sp_ind = max(superpix_img);
                    superpix_img(pos_idx(sploc_i))=max_sp_ind+1;
                end
            end
        end
    end
    superpix_img=yl_label_reranging(superpix_img)+1;
    
    if superpix_num>11
        [WeightMat,nodes_to_add_tmp] = SuperPixWeightMat_Experiment_with_Block2Block_bigdata(Superpixel_mean_Features, HyperCube, superpix_img0,superpix_img, 'CORR',conn_pattern);
%         S_W_tmp = WeightMatNormalization(WeightMat);
        S_W_tmp = (WeightMat);
    else
        WeightMat = SuperPixWeightMat_Experiment_without_Block2Block(HyperCube,superpix_img, 'CORR',conn_pattern);
%         S_W_tmp = WeightMatNormalization(WeightMat);
        S_W_tmp = (WeightMat);
    end
    
    fuse_idx=fuse_idx+1;
    WeightMatfuse{fuse_idx}=WeightMat;
    
    if isempty(S_W)
        S_W=sparse(S_W_tmp);
        nodes_to_add=nodes_to_add_tmp;
        S_W_no_normalize=WeightMat;
    else
            S_W_no_normalize = 0.5*S_W_tmp+0.5*S_W;
        %                     S_W = S_W_tmp+S_W;
        nodes_to_add=nodes_to_add|nodes_to_add_tmp;
    end
end

for train_i=1:length(TrainSamLoc)
    itrainloc=TrainSamLoc(train_i);
        sp0_idx1 = superpix_img0(itrainloc);
    itrainlabel=TrainLabels(train_i);
    otherloc=TrainSamLoc(TrainLabels==itrainlabel);
        sp0_idx2 = superpix_img0(otherloc);
        sp0_idx2 = unique(sp0_idx2);
    S_W_no_normalize(sp0_idx1,sp0_idx2)=1;
    S_W_no_normalize(sp0_idx2,sp0_idx1)=1;
    S_W_no_normalize(sp0_idx1,sp0_idx1)=0;
end