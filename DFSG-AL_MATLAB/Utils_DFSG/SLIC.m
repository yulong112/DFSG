classdef SLIC
    properties
        n_segments
        compactness
        max_iter
        min_size_factor
        max_size_factor
        sigma
        data
        labels
        segments
        superpixel_count
        S
        Q
    end
    
    methods
        function obj = SLIC(HSI, labels, n_segments, compactness, max_iter, sigma, min_size_factor, max_size_factor)
            if nargin < 8, max_size_factor = 2; end
            if nargin < 7, min_size_factor = 0.3; end
            if nargin < 6, sigma = 0; end
            if nargin < 5, max_iter = 20; end
            if nargin < 4, compactness = 20; end
            if nargin < 3, n_segments = 1000; end
            
            obj.n_segments = n_segments;
            obj.compactness = compactness;
            obj.max_iter = max_iter;
            obj.min_size_factor = min_size_factor;
            obj.max_size_factor = max_size_factor;
            obj.sigma = sigma;
            
            [height, width, bands] = size(HSI);
            data = reshape(HSI, [height * width, bands]);
%             data = zscore(data);  % 标准化
            obj.data = reshape(data, [height, width, bands]);
            obj.labels = labels;
        end
        
        function [Q, S, segments,input_img] = get_Q_and_S_and_Segments(obj)
            img = obj.data;
            [h, w, d] = size(img);
            
            % 使用 MATLAB 的 SLIC 函数
            [segments,input_img] = SLIC_fun(img, obj.n_segments, obj.compactness);
            obj.segments = segments;
            obj.superpixel_count = max(segments(:));
            disp(['superpixel_count: ', num2str(obj.superpixel_count)]);
            
            % 显示超像素边界
%             figure; imshow(boundarymask(segments));
            
            segments = reshape(segments, [], 1);
            S = zeros(obj.superpixel_count, d);
            Q = zeros(h * w, obj.superpixel_count);
            x = reshape(img, [], d);
            
            for i = 1:obj.superpixel_count
                idx = find(segments == i);
                count = length(idx);
                pixels = x(idx, :);
                superpixel = sum(pixels, 1) / count;
                S(i, :) = superpixel;
                Q(idx, i) = 1;
            end
            
            obj.S = S;
            obj.Q = Q;
        end
        
        function A = get_A(obj, sigma)
            A = zeros(obj.superpixel_count, obj.superpixel_count);
            [h, w] = size(obj.segments);
            
            for i = 1:h-2
                for j = 1:w-2
                    sub = obj.segments(i:i+1, j:j+1);
                    sub_max = max(sub(:));
                    sub_min = min(sub(:));
                    
                    if sub_max ~= sub_min
                        idx1 = sub_max;
                        idx2 = sub_min;
                        
                        if A(idx1, idx2) == 0
                            pix1 = obj.S(idx1, :);
                            pix2 = obj.S(idx2, :);
                            diss = exp(-sum((pix1 - pix2).^2) / sigma^2);
                            A(idx1, idx2) = diss;
                            A(idx2, idx1) = diss;
                        end
                    end
                end
            end
        end
    end
end
