%% load data
function WeightMat = SNG_sp_patch(ALL_data, spLocs, Height,Width)

[N,num_feature] = size(ALL_data); % k = 256, l = 1100, m = 10

ALL_data_n=ALL_data-repmat(mean(ALL_data,2),[1 size(ALL_data,2)]);
X1=abs(sqrt(sum(ALL_data_n.^2,2)));
distance= abs(ALL_data_n*ALL_data_n'./(X1*X1'));

nodes_to_retain = false(N);
for i = 1:length(spLocs)
    spLoc_i = spLocs(i);
    PointLocSet = [];
    [TempYCoord, TempXCoord] = f1DTo2DCoord([Height, Width],spLoc_i);           %计算该训练样本的坐标
    
    candidateXCoord=[TempXCoord;TempXCoord;TempXCoord-1;TempXCoord+1];
    candidateYCoord=[TempYCoord-1;TempYCoord+1;TempYCoord;TempYCoord];
    candidateLocs = (candidateXCoord-1)*Height+candidateYCoord;
    borderFilter2 = ismember(candidateLocs,spLocs);
    borderFilter=(candidateXCoord>=1)&(candidateXCoord<=Width)&(candidateYCoord>=1)&(candidateYCoord<=Height);
    candidateLocSet_tmp=candidateLocs(borderFilter&borderFilter2);
    
    [~,SNG_neighbors] = ismember(candidateLocs,spLocs);
%     [~,SNG_neighbors] = ismember(spLocs,candidateLocs);
%     connectID=find(SNG_neighbors>0);
    SNG_neighbors = SNG_neighbors(SNG_neighbors>0);
    
    nodes_to_retain(i, SNG_neighbors ) = true;
    nodes_to_retain(i,i) = false; % diagonal should be zero
end
nodes_to_retain( nodes_to_retain ~= nodes_to_retain' ) = true;
    
nodes_to_connect = false(N);
% for train_i=1:length(TrainSamLoc)
%     itrainloc=TrainSamLoc(train_i);
%     itrainlabel=TrainLabels(train_i);
%     otherloc=TrainSamLoc(TrainLabels==itrainlabel);
%     nodes_to_connect(itrainloc,otherloc)=true;
%     nodes_to_connect(otherloc,itrainloc)=true;
%     nodes_to_connect(itrainloc,itrainloc)=false;
% end

% Creating adjacency matrix: only compute values for nodes to retain
WeightMat = zeros(N);
WeightMat(nodes_to_retain) = distance(nodes_to_retain);
WeightMat(nodes_to_connect) = 1;
WeightMat = sparse(WeightMat);
