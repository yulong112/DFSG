function posterior_normalized = row_normalization(posterior_data)
    % 行归一化，确保每行和为1
    
    % 确保非负
    posterior_data = max(posterior_data, 0);
    
    row_sums = sum(posterior_data, 2);
    
    % 处理零和行
    zero_mask = row_sums == 0;
    if any(zero_mask)
        fprintf('警告: %d 行的和为0，使用均匀分布\n', sum(zero_mask));
        [n, c] = size(posterior_data);
        posterior_data(zero_mask, :) = 1/c;  % 设置为均匀分布
        row_sums(zero_mask) = 1;
    end
    
    posterior_normalized = posterior_data ./ row_sums;
    
    fprintf('行归一化后 - 每行和: min=%.6f, max=%.6f\n', ...
            min(sum(posterior_normalized, 2)), max(sum(posterior_normalized, 2)));
end