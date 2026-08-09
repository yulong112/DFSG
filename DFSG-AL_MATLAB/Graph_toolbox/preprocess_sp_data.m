%% load data
function WeightMat = preprocess_sp_data(HyperCube,superpix_img, sigma)

sp_num=max(superpix_img);
% compute pairwise distances
distance = zeros(sp_num);

[Height,Width,Bands] = size(HyperCube);
ALL_data = reshape(HyperCube, Width*Height, Bands); %3D±ä 2D

for i=1:sp_num
    for j=1:i-1
%         [~,sp_idx1]=ismember(i,superpix_img);
        sp_idx1=find(superpix_img==i);
%         [~,sp_idx2]=ismember(j,superpix_img);
        sp_idx2=find(superpix_img==j);
        sp_data_i=ALL_data(sp_idx1,:);
        sp_data_j=ALL_data(sp_idx2,:);
        X1=repmat(sum(sp_data_i.^2,2),[1 size(sp_data_j,1)]);
        X2=repmat(sum(sp_data_j.^2,2),[1 size(sp_data_i,1)]);
        R2=sp_data_i*sp_data_j';
        distance_temp = []; % norm matrix
        distance_temp = real(sqrt(X1+X2'-2*R2));
        distance(i,j) = sum(sum(distance_temp,1),2)/(size(sp_data_i,1)*size(sp_data_j,1));
%         distance(i,j) = sum(sum(distance_temp,1),2);
    end
    distance(i,i) = 0;
end
% Complete upper triangular part
distance = distance + distance';
N=sp_num;

% Number of nearest neighbors
knn_param = 10;

% Calculating distances of k-nearest neighbors
knn_distance = zeros(N,1);
for i = 1:N
    % sort all possible neighbors according to distance
    temp = sort(distance(i,:), 'ascend');
    % select k-th neighbor: knn_param+1, as the node itself is considered
    knn_distance(i) = temp(knn_param + 1);
end

% sparsification matrix
nodes_to_retain = true(N);
for i = 1:N
    nodes_to_retain(i, distance(i,:) > knn_distance(i) ) = false;
    nodes_to_retain(i,i) = false; % diagonal should be zero
end
nodes_to_retain( nodes_to_retain ~= nodes_to_retain' ) = true;

% computing sigma
sigma = 1/3 * mean(knn_distance);

% Creating adjacency matrix: only compute values for nodes to retain
WeightMat = zeros(N);
WeightMat(nodes_to_retain) = exp( -distance(nodes_to_retain).^2 / (2*sigma^2) );
WeightMat = sparse(WeightMat);
