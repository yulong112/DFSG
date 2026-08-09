%% load data
function nodes_to_retain = SuperPixel_Connect_Corr(superpix_img, Corr_Mat, flag2)

sp_num=max(superpix_img);
% compute pairwise distances
distance = zeros(sp_num);

% [Height,Width,Bands] = size(HyperCube);
% ALL_data = reshape(HyperCube, Width*Height, Bands); %3D变 2D
% %去均值处理
% ALL_data_n=ALL_data-repmat(mean(ALL_data,1),[size(ALL_data,1) 1]);
% % ALL_data_n=ALL_data-repmat(mean(ALL_data,2),[1 size(ALL_data,2)]);
% % ALL_data_n=ALL_data;
% X1=abs(sqrt(sum(ALL_data_n.^2,2)));
% %计算相关矩阵
% Corr_Mat= abs(ALL_data_n*ALL_data_n'./(X1*X1'));
% % Corr_Mat= abs(ALL_data_n*ALL_data_n');

for i=1:sp_num
    for j=1:i-1
        sp_idx1=find(superpix_img==i);
        sp_idx2=find(superpix_img==j);
        distance_temp = []; % norm matrxi
        distance_temp = Corr_Mat(sp_idx1,sp_idx2);
%         distance(i,j) = mean(mean(distance_temp(distance_temp<0.9),1),2);
%         distance(i,j) = mean(mean(distance_temp,1),2);
        distance(i,j) = max(max(distance_temp));
        %             distance(i,j_idx2) = sum(sum(distance_temp,1),2);
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
nn_distance = zeros(N,1);
for i = 1:N
    % sort all possible neighbors according to distance
    temp = sort(distance(i,:), 'descend');
    % select k-th neighbor: knn_param+1, as the node itself is considered
    knn_distance(i) = temp(knn_param + 1);
    nn_distance(i) = temp(1 + 1);
end

% sparsification matrix
nodes_to_retain = true(N);
for i = 1:N
    nodes_to_retain(i, distance(i,:) < knn_distance(i) ) = false;
    nodes_to_retain(i,i) = false; % diagonal should be zero
end
nodes_to_retain( nodes_to_retain ~= nodes_to_retain' ) = false;

% nearest neighbor matrix
nodes_to_nn = true(N);
for i = 1:N
    nodes_to_nn(i, distance(i,:) < nn_distance(i) ) = false;
    nodes_to_nn(i,i) = false; % diagonal should be zero
end
nodes_to_nn( nodes_to_nn ~= nodes_to_nn' ) = true;

if strcmp(flag2,'knn')
    nodes_to_retain = nodes_to_retain;
elseif strcmp(flag2,'nn')
    nodes_to_retain = nodes_to_nn;
elseif strcmp(flag2,'combine')
    nodes_to_retain = nodes_to_retain|nodes_to_nn;
end
