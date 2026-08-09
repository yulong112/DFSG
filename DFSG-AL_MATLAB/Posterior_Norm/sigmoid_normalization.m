function posterior_sigmoid = sigmoid_normalization(posterior_data)
    % 使用sigmoid函数进行归一化
    
    % 应用sigmoid
    posterior_sigmoid = 1 ./ (1 + exp(-posterior_data));
    
    % 行归一化
    posterior_sigmoid = posterior_sigmoid ./ sum(posterior_sigmoid, 2);
    
    fprintf('Sigmoid归一化完成\n');
end