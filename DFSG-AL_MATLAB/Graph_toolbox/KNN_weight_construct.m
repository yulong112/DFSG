%% load data
function WeightMat = KNN_weight_construct(ALL_data, sigma, param2, flag1, flag2, thita)

[N,num_feature] = size(ALL_data); % k = 256, l = 1100, m = 10
% ind=find(tril(ones(N)));
% save('ind_tmp.mat','ind')

% compute pairwise distances
tic
distance = zeros(N, N); % norm matrix
if strcmp(param2,'DIST')
    X2=repmat(sum(ALL_data.^2,2),[1 size(ALL_data,1)]);
    distance= X2+X2';
    clear X2
    R2=ALL_data*ALL_data';
    distance= distance-2*R2;
%     distance= X2+X2'-2*R2;
%     distance= abs(sqrt(X2+X2'-2*R2));
%     WeightMat(nodes_to_retain) = exp( -distance(nodes_to_retain).^2 / (2*sigma^2) );
    
elseif strcmp(param2,'DIST2')               %先去均值处理，再计算距离
    ALL_data_n=ALL_data-repmat(mean(ALL_data,2),[1 size(ALL_data,2)]);
    X2=repmat(sum(ALL_data_n.^2,2),[1 size(ALL_data_n,1)]);
%     R2=ALL_data_n*ALL_data_n';
%     distance= abs(sqrt(X2+X2'-2*R2));
%     WeightMat(nodes_to_retain) = exp( -distance(nodes_to_retain).^2 / (2*sigma^2) );
    distance= X2+X2';
    clear X2
    R2=ALL_data_n*ALL_data_n';
    distance= distance-2*R2;
%     distance= X2+X2'-2*R2;
    
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
    
elseif strcmp(param2,'CORR')
    ALL_data_n=ALL_data-repmat(mean(ALL_data,2),[1 size(ALL_data,2)]);
    clear ALL_data
    X1=abs(sqrt(sum(ALL_data_n.^2,2)));
    distance= abs(ALL_data_n*ALL_data_n'./(X1*X1'));
%     R1=X1*X1';
%     clear X1
%     R1=R1(ind);
%     save('R1_tmp.mat','R1')
%     clear R1 ind
%     distance2= ALL_data_n*ALL_data_n';
%     clear ALL_data_n
%     load('ind_tmp.mat','ind')
%     distance2=distance2(ind);
%     clear ind
%     load('R1_tmp.mat','R1')
%     distance2= abs(distance2./R1);
%     clear R1
%     distance = zeros(N, N); % norm matrix
%     load('ind_tmp.mat','ind')
%     distance(ind)=distance2;
%     clear distance2 ind
%     distance = distance + distance';
    
% elseif strcmp(param2,'CORR2')
%     distance= abs(corr(ALL_data'));
elseif strcmp(param2,'CORR2')
    distance= abs(ALL_data*ALL_data');
    
elseif strcmp(param2,'CORR3')
    X1=abs(sqrt(sum(ALL_data.^2,2)));
    distance= abs(ALL_data*ALL_data'./(X1*X1'));
    
elseif strcmp(param2,'CORR4')
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
    
elseif strcmp(param2,'Fundis')
    X2=repmat(sum(ALL_data.^2,2),[1 size(ALL_data,1)]);
    R2=ALL_data*ALL_data';
    distance1= X2+X2'-2*R2;
    
    ALL_data_n=ALL_data-repmat(mean(ALL_data,2),[1 size(ALL_data,2)]);
    X1=abs(sqrt(sum(ALL_data_n.^2,2)));
    distance2= abs(ALL_data_n*ALL_data_n'./(X1*X1'));
    
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
    
    % sparsification matrix
    nodes_to_retain = true(N);
    for i = 1:N
        if strcmp(param2,'DIST')||strcmp(param2,'DIST2')||strcmp(param2,'DIST3')
            nodes_to_retain(i, distance(i,:) > knn_distance(i) ) = false;
        else
            nodes_to_retain(i, distance(i,:) < knn_distance(i) ) = false;
        end
        nodes_to_retain(i,i) = false; % diagonal should be zero
    end
    if strcmp(flag1,'enforce')
        nodes_to_retain( nodes_to_retain ~= nodes_to_retain' ) = true;
    elseif strcmp(flag1,'mutual')
        nodes_to_retain( nodes_to_retain ~= nodes_to_retain' ) = false;
    end
    
    % nearest neighbor matrix
    nodes_to_nn = true(N);
    for i = 1:N
        if strcmp(param2,'DIST')||strcmp(param2,'DIST2')||strcmp(param2,'DIST3')
            nodes_to_nn(i, distance(i,:) > nn_distance(i) ) = false;
        else
            nodes_to_nn(i, distance(i,:) < nn_distance(i) ) = false;
        end
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
    % computing sigma
    sigma = 1/3 * mean(knn_distance);
   
% Creating adjacency matrix: only compute values for nodes to retain
WeightMat = zeros(N);
if strcmp(param2,'DIST')||strcmp(param2,'DIST2')||strcmp(param2,'DIST3')
    WeightMat(nodes_to_retain) = exp( -distance(nodes_to_retain) / (2*sigma^2) );
    
elseif strcmp(param2,'CORR')||strcmp(param2,'CORR2')||strcmp(param2,'CORR3')||strcmp(param2,'CORR4')
    WeightMat(nodes_to_retain) = distance(nodes_to_retain);
    
elseif strcmp(param2,'COVA')||strcmp(param2,'COVA2')
    WeightMat(nodes_to_retain) = distance(nodes_to_retain);
    
elseif strcmp(param2,'Fundis')
    WeightMat(nodes_to_retain) = exp( -(1-distance2(nodes_to_retain)).^2/(2*thita)-distance1(nodes_to_retain) / (2*sigma^2) );
    
end
WeightMat = sparse(WeightMat);
