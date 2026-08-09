function new_superpix_seg0 = refine_superpixels(superpix_seg0, superpix_seg1)
    % 细化超像素：根据superpix_seg1重新划分superpix_seg0中的超像素
    
    % 获取图像尺寸
    [height, width] = size(superpix_seg0);
    
    % 初始化新的超像素矩阵
    new_superpix_seg0 = zeros(height, width);
    current_label = 1;
    
    % 获取superpix_seg0中的超像素标签
    unique_labels0 = unique(superpix_seg0);
%     unique_labels0 = unique_labels0(unique_labels0 > 0); % 去除背景0
    
%     fprintf('正在处理 %d 个超像素...\n', length(unique_labels0));
    
    % 遍历superpix_seg0中的每个超像素
    for i = 1:length(unique_labels0)
        current_label0 = unique_labels0(i);
        
        % 找到当前超像素的所有像素位置
%         [rows, cols] = find(superpix_seg0 == current_label0);
%         pixel_indices = sub2ind([height, width], rows, cols);
        pixel_indices = find(superpix_seg0 == current_label0);
        
        % 查询这些像素在superpix_seg1中对应的超像素标签
        seg1_labels = superpix_seg1(pixel_indices);
        unique_seg1_labels = unique(seg1_labels);
        
        % 如果superpix_seg1中的标签种类超过1个，则进行分割
        if length(unique_seg1_labels) > 1
            % 根据superpix_seg1的标签将当前超像素分割成多个新超像素
            for j = 1:length(unique_seg1_labels)
                seg1_label = unique_seg1_labels(j);
                
                % 找到在当前超像素中属于该seg1标签的像素
                mask_in_current = (superpix_seg1 == seg1_label) & (superpix_seg0 == current_label0);
                
                % 如果这个区域不为空，则分配新标签
                if any(mask_in_current(:))
                    new_superpix_seg0(mask_in_current) = current_label;
                    current_label = current_label + 1;
                end
            end
        else
            % 如果只有一种标签，保持原样
            new_superpix_seg0(superpix_seg0 == current_label0) = current_label;
            current_label = current_label + 1;
        end
        
%         % 显示进度
%         if mod(i, 100) == 0
%             fprintf('已处理 %d/%d 个超像素\n', i, length(unique_labels0));
%         end
    end
    
%     fprintf('处理完成！新的超像素数量: %d\n', current_label - 1);
end