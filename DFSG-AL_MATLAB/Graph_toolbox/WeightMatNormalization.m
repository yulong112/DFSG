function S_W = WeightMatNormalization(WeightMat)
N = size(WeightMat,1);
d = sum(WeightMat,2);
d(d~=0) = d(d~=0).^(-1/2);
Dinv = spdiags(d,0,N,N);
S_W = Dinv*WeightMat*Dinv;
clear Dinv WeightMat;
S_W=sparse(S_W);