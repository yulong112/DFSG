function S_W_no_normalize = FSPG_graph_construction_full(HyperCube,superpix_allnum,conn_pattern,TrainSamLoc,TrainLabels)
[Height,Width,Bands] = size(HyperCube);
HyperCube_vec = reshape(HyperCube, Width*Height, Bands); %3D鍙?2D
S_W=0;
WeightMatfuse={};
fuse_idx=0;
for superpix_num=superpix_allnum
    %% 超像素谱图聚类。
    superpix_img = superpixel_cut(HyperCube,superpix_num);
    sp_num = max(superpix_img);
    for select_i=1:sp_num
        pos_idx=find(superpix_img==select_i);
        spTrainflag=ismember(TrainSamLoc,pos_idx);
        spTrainSamLoc = TrainSamLoc(spTrainflag);
        spTrainLabels = TrainLabels(spTrainflag);
        if length(spTrainSamLoc)
            spTrainLabels=yl_label_reranging(spTrainLabels)+1;
            if max(spTrainLabels)>1
                % 瓒呰繃涓ょ被锛屽垎鍓茶秴鍍忕礌涓哄鍧?
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
                para_alpha=0.1;
                UU = speye(N_sp) - para_alpha * SW_sp;
                F = UU \ initial_Labels_sp; % classification function F = (I - \alpha S)^{-1}Y
                [~, PredLabels_sp] = max(F, [], 2); %simply checking which of elements is largest in each row
                % 灏嗗叾涓殑鏌愪簺瓒呭儚绱犲簭鍙峰鍔?
                for splabel_i=2:max(PredLabels_sp)
                    sploc_i = find(PredLabels_sp==splabel_i);
                    max_sp_ind = max(superpix_img);
                    superpix_img(pos_idx(sploc_i))=max_sp_ind+1;
                end
            end
        end
    end
    superpix_img=yl_label_reranging(superpix_img)+1;
    
    %% superpixel segmentation Error
%     counts_error=0;
%     for spi=1:max(superpix_img)
%         TMtmp = TruthMap1D(superpix_img==spi);
%         if length(unique(TMtmp(TMtmp~=0)))>1
%             counts_error=counts_error+1;
%         end
%     end
%     ER = double(counts_error)/double(max(superpix_img))*100
    
    if superpix_num>11
        [WeightMat,nodes_to_add_tmp] = SuperPixWeightMat_Experiment_with_Block2Block(HyperCube,superpix_img, 'CORR',conn_pattern);
        S_W_tmp = WeightMatNormalization(WeightMat);
    else
        WeightMat = SuperPixWeightMat_Experiment_without_Block2Block(HyperCube,superpix_img, 'CORR',conn_pattern);
        S_W_tmp = WeightMatNormalization(WeightMat);
    end
    
    fuse_idx=fuse_idx+1;
    WeightMatfuse{fuse_idx}=S_W_tmp;
    S_W = S_W+WeightMatfuse{fuse_idx};
end

N = size(WeightMat,1);
tri_ind=find(tril(ones(N)));
S_W_tri=S_W(tri_ind);
non_ind = find(S_W_tri);
clear S_W

OriginWeightMat=[];

for idx=1:fuse_idx
    S_W1=WeightMatfuse{idx};
    S_W1_tri=S_W1(tri_ind);
    S_W1_tri_non = S_W1_tri(non_ind);
    OriginWeightMat=[OriginWeightMat S_W1_tri_non];
end

zeros_len=20*size(OriginWeightMat,1);
[PCAWeightMat,PCA_direction] = fPCA_FinalVersionV2([zeros(zeros_len,size(OriginWeightMat,2));full(OriginWeightMat)],1,0,0);
PCAWeightMat=PCAWeightMat(zeros_len+1:end,:);
% [PCAWeightMat,PCA_direction] = fPCA_FinalVersionV2(full(OriginWeightMat),1,0,0);

tri_vector = zeros(length(tri_ind),1);
tri_vector(non_ind) = abs(PCAWeightMat);
S_W = sparse(zeros(N, N)); % norm matrix
S_W(tri_ind)=tri_vector;
S_W = S_W + S_W';

clear OriginWeightMat PCAWeightMat TempS_W
d = sum(S_W,2);
d(d~=0) = d(d~=0).^(-1/2);
Dinv = spdiags(d,0,N,N);

S_W_no_normalize = S_W;
%     S_W_no_normalize = Dinv*S_W*Dinv;



for train_i=1:length(TrainSamLoc)
    itrainloc=TrainSamLoc(train_i);
    itrainlabel=TrainLabels(train_i);
    otherloc=TrainSamLoc(TrainLabels==itrainlabel);
    S_W_no_normalize(itrainloc,otherloc)=1;
    S_W_no_normalize(otherloc,itrainloc)=1;
    S_W_no_normalize(itrainloc,itrainloc)=0;
end