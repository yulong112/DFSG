function posterior_softmax = softmax_normalization(posterior_data)
    % 标准的softmax归一化
    % posterior_data: N*c 矩阵
    
    % 减去最大值提高数值稳定性
    max_vals = max(posterior_data, [], 2);
    exp_data = exp(posterior_data - max_vals);
    
    % 计算softmax
    sum_exp = sum(exp_data, 2);
    posterior_softmax = exp_data ./ sum_exp;
    
    fprintf('Softmax后 - 每行和: min=%.6f, max=%.6f\n', ...
            min(sum(posterior_softmax, 2)), max(sum(posterior_softmax, 2)));
end