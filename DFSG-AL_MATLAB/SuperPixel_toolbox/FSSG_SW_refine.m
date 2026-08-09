%% load data
function [S_W_refined,nodes_to_refine] = FSSG_SW_refine(S_W,nodes_to_add, flag1, flag2)

nodes_to_add_flags=sum(nodes_to_add);
frontsceneID=find(nodes_to_add_flags>0);

    % Number of nearest neighbors
    knn_param = 10;
    % Calculating distances of k-nearest neighbors
    knn_similarity = zeros(length(frontsceneID),1);
    nn_similarity = zeros(length(frontsceneID),1);
    
    KNN_refind_id=0;
    
    for frontsceneID_i=frontsceneID
        connection_SG=S_W(frontsceneID_i,frontsceneID);
        % sort all possible neighbors according to similarity
        temp = sort(connection_SG, 'descend');
        
        KNN_refind_id=KNN_refind_id+1;
        % select k-th neighbor: knn_param+1, as the node itself is considered
        knn_similarity(KNN_refind_id) = temp(knn_param + 1);
        nn_similarity(KNN_refind_id) = temp(3);
    end
    
    % sparsification matrix
    nodes_to_refine = sparse([]);
    KNN_refind_id=0;
    for frontsceneID_i=frontsceneID
        KNN_refind_id=KNN_refind_id+1;
            nodes_to_refine(frontsceneID_i, S_W(frontsceneID_i,:) > knn_similarity(KNN_refind_id) ) = true;
    end
    Mat_N=size(S_W,1);
    [Rows,Cols]=size(nodes_to_refine);
    if Rows<Mat_N
        nodes_to_refine(Rows+1:Mat_N,:)=false(Mat_N-Rows,Cols);
    end
    if Cols<Mat_N
        nodes_to_refine(:,Cols+1:Mat_N)=false(Mat_N,Mat_N-Cols);
    end
    
    if strcmp(flag1,'enforce')
        nodes_to_refine( nodes_to_refine ~= nodes_to_refine' ) = true;
    elseif strcmp(flag1,'mutual')
        nodes_to_refine( nodes_to_refine ~= nodes_to_refine' ) = false;
    end
    
    % nearest neighbor matrix
    nodes_to_nn = sparse([]);
    KNN_refind_id=0;
    for frontsceneID_i=frontsceneID
        KNN_refind_id=KNN_refind_id+1;
            nodes_to_nn(frontsceneID_i, S_W(frontsceneID_i,:) > nn_similarity(KNN_refind_id) ) = true;
    end
    [Rows,Cols]=size(nodes_to_nn);
    if Rows<Mat_N
        nodes_to_nn(Rows+1:Mat_N,:)=false(Mat_N-Rows,Cols);
    end
    if Cols<Mat_N
        nodes_to_nn(:,Cols+1:Mat_N)=false(Mat_N,Mat_N-Cols);
    end
    nodes_to_nn( nodes_to_nn ~= nodes_to_nn' ) = true;
    
    if strcmp(flag2,'knn')
        nodes_to_refine = nodes_to_refine;
    elseif strcmp(flag2,'nn')
        nodes_to_refine = nodes_to_nn;
    elseif strcmp(flag2,'combine')
        nodes_to_refine = nodes_to_refine|nodes_to_nn;
    end
    
    S_W_refined=S_W;
    S_W_refined(nodes_to_add)=0;
    S_W_refined(nodes_to_refine~=0)=S_W(nodes_to_refine~=0);
