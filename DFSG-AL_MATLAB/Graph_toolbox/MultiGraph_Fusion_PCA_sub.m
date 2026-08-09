function [S_W,S_W_no_normalize] = MultiGraph_Fusion_PCA_sub(S_W1, S_W2, flag_normalize)

Laplace_Mat=0;

if ~exist('flag_normalize','var')
    flag_normalize=1;
end

[SW_N]=size(S_W1,1);
MultiLayerWeight=[];
ind=0;
coordinate_mat=[];
S_W1=triu(S_W1);
ind_SW1=find(S_W1~=0);
S_W2=triu(S_W2);
ind_SW2=find(S_W2~=0);
ind_SW=union(ind_SW1, ind_SW2);

vecSW1=S_W1(ind_SW);
clear S_W1

vecSW2=S_W2(ind_SW);
clear S_W2

MultiLayerWeight=[full(vecSW1) full(vecSW2)];
clear vecSW1 vecSW2

% for i=1:SW_N
%     for j=1:i
%         ind=ind+1;
%         MultiLayerWeight(ind,1)=S_W1(i,j);
%         MultiLayerWeight(ind,2)=S_W2(i,j);
%         coordinate_mat(ind,1)=i;
%         coordinate_mat(ind,2)=j;
%     end
% end
% clear S_W1 S_W2

PCAWeightMat = fPCA_FinalVersionV2(MultiLayerWeight,1,0,0);
S_W=zeros(SW_N);
S_W(ind_SW)=abs(PCAWeightMat);
% for ind=1:length(PCAWeightMat)
%         S_W(coordinate_mat(ind,:))=abs(PCAWeightMat(ind));
% end
clear PCAWeightMat
S_W=S_W+S_W';
S_W=sparse(S_W);
S_W_no_normalize=S_W;

if flag_normalize==1
    N = size(S_W,1);
    d = sum(S_W,2);
    d(d~=0) = d(d~=0).^(-1/2);
    Dinv = spdiags(d,0,N,N);
    S_W = Dinv*S_W*Dinv;
    fprintf('execute normalization of SW mat\n');
end