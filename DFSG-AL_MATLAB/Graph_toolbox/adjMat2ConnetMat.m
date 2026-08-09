%% load data
function nodes_to_retain = adjMat2ConnetMat(distance, flag2)

N=size(distance,1);

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
