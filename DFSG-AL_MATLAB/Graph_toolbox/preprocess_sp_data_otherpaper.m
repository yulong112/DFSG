%% load data
function [WeightMat,nodes_to_retain] = preprocess_sp_data_otherpaper(HyperCube,superpix_img, sigma_s, sigma_l, kernel_h, weight_beta)
if ~exist('kernel_h','var')
    kernel_h = 15;
end
if ~exist('weight_beta','var')
    weight_beta = 0.5;
end

sp_num=max(superpix_img);
% compute pairwise distances
distance = zeros(sp_num);

[Height,Width,Bands] = size(HyperCube);
ALL_data = reshape(HyperCube, Width*Height, Bands); %3D±ä 2D

Superpixel_Mean_Features=[];
for i=1:sp_num
    sp_idx1= superpix_img==i;
    sp_data_i=ALL_data(sp_idx1,:);
    X1=mean(sp_data_i,1);
    Superpixel_Mean_Features=[Superpixel_Mean_Features;X1];
end

Superpixel_Weighted_Features=[];
Superpixel_centroidal_Loc=[];
for i=1:sp_num
    sp_idx1=find(superpix_img==i);
    [Superpixel_Weighted_Feature, centroidal_Loc] = ...
        Superpixel_Weighted_Features_Compute(i,sp_idx1,superpix_img,Height,Width,Superpixel_Mean_Features,kernel_h);

    Superpixel_Weighted_Features=[Superpixel_Weighted_Features;Superpixel_Weighted_Feature];
    Superpixel_centroidal_Loc=[Superpixel_centroidal_Loc;centroidal_Loc];
end

    X2=repmat(sum(Superpixel_Mean_Features.^2,2),[1 size(Superpixel_Mean_Features,1)]);
    distance1= X2+X2';
    clear X2
    R2=Superpixel_Mean_Features*Superpixel_Mean_Features';
    distance1= distance1-2*R2;

    X2=repmat(sum(Superpixel_Weighted_Features.^2,2),[1 size(Superpixel_Weighted_Features,1)]);
    distance2= X2+X2';
    clear X2
    R2=Superpixel_Weighted_Features*Superpixel_Weighted_Features';
    distance2= distance2-2*R2;
    
    distance12=(1-weight_beta)*distance1+weight_beta*distance2;
    
% for i=1:sp_num
%         sp_idx1=find(superpix_img==i);
%         X1=Superpixel_Mean_Features(i,:);
%     for j=1:i-1
% %         [~,sp_idx1]=ismember(i,superpix_img);
% %         [~,sp_idx2]=ismember(j,superpix_img);
% %         sp_data_i=ALL_data(sp_idx1,:);
% %         sp_data_j=ALL_data(sp_idx2,:);
% %         X2=mean(sp_data_j);
%         sp_idx2=find(superpix_img==j);
%         X2=Superpixel_Mean_Features(j,:);
%         
%         distance(i,j) = sum((X1-X2).^2,2);
% %         distance(i,j) = sum(sum(distance_temp,1),2);
%     end
%     distance(i,i) = 0;
% end

% Complete upper triangular part
N=sp_num;

% Number of nearest neighbors
knn_param = 8;

% Calculating distances of k-nearest neighbors
knn_distance = zeros(N,1);
for i = 1:N
    % sort all possible neighbors according to distance
    temp = sort(distance12(i,:), 'ascend');
    % select k-th neighbor: knn_param+1, as the node itself is considered
    knn_distance(i) = temp(knn_param + 1);
end
% computing sigma_s
sigma_s = 1/3 * mean(knn_distance);

weight_component1=exp( -distance12 / (sigma_s^2) );


    X2=repmat(sum(Superpixel_centroidal_Loc.^2,2),[1 size(Superpixel_centroidal_Loc,1)]);
    distance3= X2+X2';
    clear X2
    R2=Superpixel_centroidal_Loc*Superpixel_centroidal_Loc';
    distance3= distance3-2*R2;
    weight_component2=exp( -distance3 / (sigma_l^2) );
    
    weight_composite=weight_component1.*weight_component2;

% Calculating distances of k-nearest neighbors
knn_distance = zeros(N,1);
for i = 1:N
    % sort all possible neighbors according to distance
    temp = sort(weight_composite(i,:), 'descend');
    % select k-th neighbor: knn_param+1, as the node itself is considered
    knn_distance(i) = temp(knn_param + 1);
end
% sparsification matrix
nodes_to_retain = true(N);
for i = 1:N
    nodes_to_retain(i, weight_composite(i,:) < knn_distance(i) ) = false;
    nodes_to_retain(i,i) = false; % diagonal should be zero
end
nodes_to_retain( nodes_to_retain ~= nodes_to_retain' ) = true;

% Creating adjacency matrix: only compute values for nodes to retain
WeightMat = zeros(N);
WeightMat(nodes_to_retain) = weight_composite(nodes_to_retain);
WeightMat = sparse(WeightMat);
