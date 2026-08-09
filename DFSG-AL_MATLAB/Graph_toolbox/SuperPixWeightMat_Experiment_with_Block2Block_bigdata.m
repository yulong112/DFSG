%% load data
function [WeightMat,nodes_to_add] = SuperPixWeightMat_Experiment_with_Block2Block_bigdata(Superpixel_mean_Features, HyperCube, superpix_img0, superpix_img, param2, flag2)

sp_num0=max(superpix_img0);
sp_num=max(superpix_img);
[Height,Width,Bands] = size(HyperCube);
ALL_data = reshape(HyperCube, Width*Height, Bands); %3D变 2D

Superpixel_mean_Features1=[];
for i=1:sp_num
    sp_idx1=find(superpix_img==i);
    X1=mean(ALL_data(sp_idx1,:),1);
    Superpixel_mean_Features1=[Superpixel_mean_Features1;X1];
end

% % compute pairwise distances
% [Height,Width,Bands] = size(HyperCube);
% ALL_data = reshape(HyperCube, Width*Height, Bands); %3D变 2D
% 
% Superpixel_mean_Features=[];
% for i=1:sp_num0
%     sp_idx1=find(superpix_img0==i);
%     X1=mean(ALL_data(sp_idx1,:),1);
%     Superpixel_mean_Features=[Superpixel_mean_Features;X1];
% end
ALL_data = Superpixel_mean_Features;
clear Superpixel_mean_Features
[N,num_feature] = size(ALL_data); % k = 256, l = 1100, m = 10
% compute pairwise distances
tic
distance = zeros(N, N); % norm matrix
if strcmp(param2,'DIST')
    X2=repmat(sum(ALL_data.^2,2),[1 size(ALL_data,1)]);
    R2=ALL_data*ALL_data';
%     distance= abs(sqrt(X2+X2'-2*R2));
%     WeightMat(nodes_to_retain) = exp( -distance(nodes_to_retain).^2 / (2*sigma^2) );
    distance= X2+X2'-2*R2;
    
elseif strcmp(param2,'DIST2')               %先去均值处理，再计算距离
    ALL_data_n=ALL_data-repmat(mean(ALL_data,2),[1 size(ALL_data,2)]);
    X2=repmat(sum(ALL_data_n.^2,2),[1 size(ALL_data_n,1)]);
    R2=ALL_data_n*ALL_data_n';
%     distance= abs(sqrt(X2+X2'-2*R2));
%     WeightMat(nodes_to_retain) = exp( -distance(nodes_to_retain).^2 / (2*sigma^2) );
    distance= X2+X2'-2*R2;
    
elseif strcmp(param2,'DIST3')
%     ALL_data=ALL_data-repmat(mean(ALL_data,1),[size(ALL_data,1) 1]);
    % compute pairwise distances
    distance = zeros(N);
    % Assign lower triangular part only in loop, saves time
    for i = 1:N
        for j = 1:i-1
            distance(i,j) = sum(abs(ALL_data(i,:) - ALL_data(j,:)));
        end
    end
    % Complete upper triangular part
    distance = distance + distance';
    distance = distance.^2;
    
elseif strcmp(param2,'CORR')  %Feature Mean-Residual Normalized Correlation
    ALL_data_n=ALL_data-repmat(mean(ALL_data,2),[1 size(ALL_data,2)]);
    X1=abs(sqrt(sum(ALL_data_n.^2,2)));
    X1(X1==0)=1;
    distance= abs(ALL_data_n*ALL_data_n'./(X1*X1'));
    
    ALL_data_n=Superpixel_mean_Features1-repmat(mean(Superpixel_mean_Features1,2),[1 size(Superpixel_mean_Features1,2)]);
    X1=abs(sqrt(sum(ALL_data_n.^2,2)));
    X1(X1==0)=1;
    distance1= abs(ALL_data_n*ALL_data_n'./(X1*X1'));
    
% elseif strcmp(param2,'CORR2')
%     distance= abs(corr(ALL_data'));
elseif strcmp(param2,'CORR2')
    distance= abs(ALL_data*ALL_data');
    
elseif strcmp(param2,'CORR3')
    X1=abs(sqrt(sum(ALL_data.^2,2)));
    distance= abs(ALL_data*ALL_data'./(X1*X1'));
    
elseif strcmp(param2,'CORR4')  %Sample Mean-Residual Normalized Correlation
    ALL_data_n=ALL_data-repmat(mean(ALL_data,1),[size(ALL_data,1) 1]);
    X1=abs(sqrt(sum(ALL_data_n.^2,2)));
    distance= abs(ALL_data_n*ALL_data_n'./(X1*X1'));
    
elseif strcmp(param2,'COVA')
%     distance= abs(cov(ALL_data'));
    ALL_data_n=ALL_data-repmat(mean(ALL_data,1),[size(ALL_data,1) 1]);
%     X1=abs(sqrt(sum(ALL_data_n.^2,2)));
    distance= abs(ALL_data_n*ALL_data_n'/(num_feature-1));
    
elseif strcmp(param2,'COVA2')
    distance= abs(cov(ALL_data'));
    
end
compute_distance_time=toc

    % Number of nearest neighbors
    knn_param = 10;
    % Calculating distances of k-nearest neighbors
    knn_distance = zeros(N,1);
    nn_distance = zeros(N,1);
    for i = 1:N
        % sort all possible neighbors according to distance
        if strcmp(param2,'DIST')||strcmp(param2,'DIST2')||strcmp(param2,'DIST3')
            temp = sort(distance(i,:), 'ascend');
        else
            temp = sort(distance(i,:), 'descend');
        end
        % select k-th neighbor: knn_param+1, as the node itself is considered
        knn_distance(i) = temp(knn_param + 1);
        nn_distance(i) = temp(2);
    end
    
    % computing sigma
    sigma = 1/3 * mean(knn_distance);
    
    %% 以下是基于超像素的连接策略。
    %%%%%%%同一超像素块内使用SNN连接，不同超像素块之间先不连接%%%%%%
    nodes_to_retain = false(N);
    for i=1:sp_num
        sp_idx1 = find(superpix_img==i);
        
        for idxii = 1:length(sp_idx1)
            pixel_idx = sp_idx1(idxii);
            sp0_idx = superpix_img0(pixel_idx);
            PointLocSet = [];
            [TempYCoord, TempXCoord] = f1DTo2DCoord([Height, Width],pixel_idx);           %计算该训练样本的坐标
            
            %分三种情况：图像下边界，图像上边界，图像中间其他位置
            if mod(pixel_idx,Height)==0
                lower_y=-1;upper_y=0;
            elseif mod(pixel_idx-1,Height)==0
                lower_y=0;upper_y=1;
            else
                lower_y=-1;upper_y=1;
            end
            
            for y_j = lower_y:upper_y
                PointLocTemp = (TempXCoord-1)*Height+TempYCoord+y_j;
                if ismember(PointLocTemp,sp_idx1)
                    SP0LocTemp = superpix_img0(PointLocTemp);
                    SP0LocTemp = unique(SP0LocTemp);
                    SP0LocTemp = SP0LocTemp(SP0LocTemp~=sp0_idx);
                    PointLocSet=[PointLocSet;SP0LocTemp];
                end
            end
            
            for x_i = -1:1
                PointLocTemp = (TempXCoord-1+x_i)*Height+TempYCoord;
                if ismember(PointLocTemp,sp_idx1)
                    SP0LocTemp = superpix_img0(PointLocTemp);
                    SP0LocTemp = unique(SP0LocTemp);
                    SP0LocTemp = SP0LocTemp(SP0LocTemp~=sp0_idx);
                    PointLocSet=[PointLocSet;SP0LocTemp];
                end
            end
            nodes_to_retain( sp0_idx, PointLocSet ) = true;
            nodes_to_retain( sp0_idx, sp0_idx ) = false; % diagonal should be zero
        end
        
    end
    nodes_to_retain( nodes_to_retain ~= nodes_to_retain' ) = true;
    nodes_to_redefine=nodes_to_retain;
    
    %%%%%%%不同超像素块之间使用互K近邻+强制最近邻连接%%%%%%
    nodes_to_super=[];
    if strcmp(param2,'DIST')||strcmp(param2,'DIST2')||strcmp(param2,'DIST3')
%         nodes_to_super = SuperPixel_Connect_Distance(HyperCube, superpix_img, flag2);
    else
        nodes_to_super = adjMat2ConnetMat(distance1, flag2);
%         nodes_to_super = SuperPixel_Connect_Corr_bigdata(superpix_img0, superpix_img, distance, flag2);
    end
    
    %%%%%%%将有连接的超像素块里的点之间使用强制最近邻连接%%%%%%
%     nodes_to_add = Connect_pixel_btwsp(nodes_to_super,superpix_img, distance, param2, flag2);
    nodes_to_add = [];
    nodes_to_add = Connect_pixel_btwsp2_bigdata(nodes_to_super,superpix_img0,superpix_img, distance, param2);
    nodes_to_retain = nodes_to_retain|nodes_to_add;
    nodes_to_retain( nodes_to_retain ~= nodes_to_retain' ) = true;
    
    nodes_to_add=sparse(nodes_to_add);
    
% Creating adjacency matrix: only compute values for nodes to retain
WeightMat = zeros(N);
if strcmp(param2,'DIST')||strcmp(param2,'DIST2')||strcmp(param2,'DIST3')
    WeightMat(nodes_to_retain) = exp( -distance(nodes_to_retain) / (2*sigma^2) );
    WeightMat(nodes_to_redefine) = 2;
    
elseif strcmp(param2,'CORR')||strcmp(param2,'CORR2')||strcmp(param2,'CORR3')||strcmp(param2,'CORR4')||strcmp(param2,'CORR5')
    WeightMat(nodes_to_retain) = distance(nodes_to_retain);
%     WeightMat(nodes_to_add) = 0.1*WeightMat(nodes_to_add);
%     max_similar=ceil(max(max(distance)));
%     WeightMat(nodes_to_redefine) = 2*max_similar;
    
elseif strcmp(param2,'COVA')||strcmp(param2,'COVA2')
    WeightMat(nodes_to_retain) = distance(nodes_to_retain);
    max_similar=ceil(max(max(distance)));
    WeightMat(nodes_to_redefine) = 2*max_similar;
    
elseif strcmp(param2,'Fundis')
    WeightMat(nodes_to_retain) = exp( -(1-distance2(nodes_to_retain)).^2/(2*thita)-distance1(nodes_to_retain) / (2*sigma^2) );
    
end
WeightMat = sparse(WeightMat);

