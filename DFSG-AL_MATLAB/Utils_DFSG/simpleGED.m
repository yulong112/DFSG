function ged = simpleGED(adj1, adj2, node_labels1, node_labels2)
% 简化的图编辑距离计算
% adj1, adj2: 邻接矩阵
% node_labels1, node_labels2: 节点标签（可选）

    if nargin < 3
        node_labels1 = 1:size(adj1,1);
        node_labels2 = 1:size(adj2,1);
    end
    
    n1 = size(adj1, 1);
    n2 = size(adj2, 1);
    max_nodes = max(n1, n2);
    
    % 扩展邻接矩阵到相同大小
    adj1_ext = zeros(max_nodes);
    adj2_ext = zeros(max_nodes);
    
    adj1_ext(1:n1, 1:n1) = adj1;
    adj2_ext(1:n2, 1:n2) = adj2;
    
    % 生成所有可能的排列
    min_cost = inf;
    
    if max_nodes <= 6
        % 对于小图，尝试所有排列
        perms = perms(1:max_nodes);
        for i = 1:size(perms, 1)
            p = perms(i, :);
            adj2_perm = adj2_ext(p, p);
            
            % 计算编辑成本
            node_cost = calculateNodeCost(node_labels1, node_labels2, p, n1, n2);
            edge_cost = sum(sum(abs(adj1_ext - adj2_perm))) / 2; % 无向图除以2
            
            total_cost = node_cost + edge_cost;
            
            if total_cost < min_cost
                min_cost = total_cost;
            end
        end
    else
        % 对于大图，使用近似方法
        min_cost = approximateGED(adj1, adj2, node_labels1, node_labels2);
    end
    
    ged = min_cost;
end

function ged = approximateGED(adj1, adj2, node_labels1, node_labels2)
% 近似图编辑距离计算
    if nargin < 3
        node_labels1 = 1:size(adj1,1);
        node_labels2 = 1:size(adj2,1);
    end
    
    n1 = size(adj1, 1);
    n2 = size(adj2, 1);
    
    % 方法1: 基于度序列的近似
    deg1 = sum(adj1, 2)';
    deg2 = sum(adj2, 2)';
    
    % 归一化度序列
    max_deg1 = max(deg1); max_deg2 = max(deg2);
    if max_deg1 > 0, deg1 = deg1 / max_deg1; end
    if max_deg2 > 0, deg2 = deg2 / max_deg2; end
    
    % 填充到相同长度
    max_len = max(length(deg1), length(deg2));
    deg1_padded = zeros(1, max_len);
    deg2_padded = zeros(1, max_len);
    
    deg1_padded(1:length(deg1)) = deg1;
    deg2_padded(1:length(deg2)) = deg2;
    
    % 计算度序列差异
    degree_diff = sum(abs(deg1_padded - deg2_padded));
    
    % 方法2: 基于特征值的近似
    try
        if n1 > 0
            eigs1 = sort(eig(adj1), 'descend');
        else
            eigs1 = [];
        end
        if n2 > 0
            eigs2 = sort(eig(adj2), 'descend');
        else
            eigs2 = [];
        end
        
        % 填充特征值
        max_eig_len = max(length(eigs1), length(eigs2));
        eigs1_padded = zeros(1, max_eig_len);
        eigs2_padded = zeros(1, max_eig_len);
        
        eigs1_padded(1:length(eigs1)) = eigs1;
        eigs2_padded(1:length(eigs2)) = eigs2;
        
        eigen_diff = sum(abs(eigs1_padded - eigs2_padded));
    catch
        eigen_diff = 0;
    end
    
    % 节点数量差异
    size_penalty = abs(n1 - n2);
    
    % 组合所有度量
    ged = degree_diff + 0.5 * eigen_diff + 0.3 * size_penalty;
end

function node_cost = calculateNodeCost(labels1, labels2, permutation, n1, n2)
% 计算节点编辑成本
    node_cost = 0;
    max_nodes = length(permutation);
    
    for i = 1:max_nodes
        if i <= n1 && permutation(i) <= n2
            % 节点替换
            if labels1(i) ~= labels2(permutation(i))
                node_cost = node_cost + 1;
            end
        elseif i <= n1
            % 节点删除
            node_cost = node_cost + 1;
        else
            % 节点插入
            node_cost = node_cost + 1;
        end
    end
end