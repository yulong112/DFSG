function S_W = MultiGraph_Fusion_PCA(S_W1, S_W2)

TempS_W=cat(3,full(S_W1),full(S_W2));
[N,N,Levels] = size(TempS_W);
OriginWeightMat = reshape(TempS_W,N*N,Levels);
PCAWeightMat = fPCA_FinalVersionV2(OriginWeightMat,1,0,0);
PCAWeightMat = reshape(PCAWeightMat,N,N);
S_W = abs(PCAWeightMat);
S_W=sparse(S_W);
N = size(S_W,1);
d = sum(S_W,2);
d(d~=0) = d(d~=0).^(-1/2);
Dinv = spdiags(d,0,N,N);
S_W = Dinv*S_W*Dinv;