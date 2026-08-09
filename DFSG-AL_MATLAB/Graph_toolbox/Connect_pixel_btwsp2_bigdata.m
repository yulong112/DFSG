function nodes_to_add = Connect_pixel_btwsp2_bigdata(nodes_to_super,superpix_img0,superpix_img, distance, param2)
N=size(distance,1);
nodes_to_add = false(N);
sp_num=size(nodes_to_super,1);
center_set=zeros(sp_num,1);

% method1: max connectivity
for i = 1:sp_num
    pixel_set = find(superpix_img==i);
    sp0_idx = superpix_img0(pixel_set);
    sp0_idx = unique(sp0_idx);
    tmp_distance=distance(sp0_idx,sp0_idx);
    % Number of nearest neighbors
    knn_param = 10;
    % Calculating distances of k-nearest neighbors
    knn_distance = zeros(length(sp0_idx),1);
    for kk = 1:length(sp0_idx)
        % sort all possible neighbors according to distance
        if strcmp(param2,'DIST')||strcmp(param2,'DIST2')||strcmp(param2,'DIST3')
            temp = sort(distance(kk,:), 'ascend');
        else
%             temp = sort(distance(kk,:), 'descend');
            temp = sort(distance(sp0_idx(kk),:), 'descend');
        end
        % select k-th neighbor: knn_param+1, as the node itself is considered
        knn_distance(kk) = temp(knn_param + 1);
    end
    nodes_to_knn = true(length(sp0_idx));
    for kk = 1:length(sp0_idx)
        if strcmp(param2,'DIST')||strcmp(param2,'DIST2')||strcmp(param2,'DIST3')
            nodes_to_knn(kk, tmp_distance(kk,:) > knn_distance(kk) ) = false;
        else
            nodes_to_knn(kk, tmp_distance(kk,:) < knn_distance(kk) ) = false;
        end
        nodes_to_knn(kk,kk) = false; % diagonal should be zero
    end
    [~,center_point]=max(sum(nodes_to_knn));
    center_set(i)=sp0_idx(center_point);
end

% method2: max weights
% for i = 1:sp_num
%     pixel_set = find(superpix_img==i);
%     sp0_idx = superpix_img0(pixel_set);
%     sp0_idx = unique(sp0_idx);
%     tmp_distance=distance(sp0_idx,sp0_idx);
%     [~,center_point]=max(sum(tmp_distance));
%     center_set(i)=sp0_idx(center_point);
% end

% method3: rand
% rng('default')
% seed = 10;
% seed = 2;
% seed = 50;
% rng(seed);
% for i = 1:sp_num
%     pixel_set = find(superpix_img==i);
%     sp0_idx = superpix_img0(pixel_set);
%     sp0_idx = unique(sp0_idx);
%     center_point=randperm(length(sp0_idx),1);
%     center_set(i)=sp0_idx(center_point);
% end

for i = 1:sp_num
    similar_set=find(nodes_to_super(i,:)>0);
    similar_set=[similar_set i];
    
    if length(similar_set)>1
        for i_set=2:length(similar_set)
            for j_set=1:i_set-1
                point_idx1=center_set(similar_set(i_set));
                point_idx2=center_set(similar_set(j_set));
                nodes_to_add(point_idx1,point_idx2)=true;
                nodes_to_add(point_idx2,point_idx1)=true;
            end
        end
        
    else
        printtex=['³öÏÖ¹ÂÁ¢³¬ÏñËØ¿é,³¬ÏñËØ¿éĞòºÅ£º',num2str(i)]
    end
end
for i=1:N
    nodes_to_add(i,i)=false;
end
nodes_to_add( nodes_to_add ~= nodes_to_add' ) = true;
