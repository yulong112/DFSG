
function [f1_mean, f1_per_class, precision_per_class, recall_per_class] = calculate_f1_score(truth_valid, pred_valid)
% 初始化每个类别的统计量
num_classes = length(unique(truth_valid(truth_valid>0)));
tp_per_class = zeros(num_classes, 1);  % 真正例
fp_per_class = zeros(num_classes, 1);  % 假正例  
fn_per_class = zeros(num_classes, 1);  % 假反例

% 计算每个类别的TP, FP, FN
for class_idx = 1:num_classes
    % 真正例：预测为当前类且真实为当前类
    tp = sum((pred_valid == class_idx) & (truth_valid == class_idx));
    
    % 假正例：预测为当前类但真实不是当前类
    fp = sum((pred_valid == class_idx) & (truth_valid ~= class_idx));
    
    % 假反例：真实为当前类但预测不是当前类
    fn = sum((pred_valid ~= class_idx) & (truth_valid == class_idx));
    
    tp_per_class(class_idx) = tp;
    fp_per_class(class_idx) = fp;
    fn_per_class(class_idx) = fn;
end

% 计算每个类别的精确率和召回率
precision_per_class = zeros(num_classes, 1);
recall_per_class = zeros(num_classes, 1);
f1_per_class = zeros(num_classes, 1);

for class_idx = 1:num_classes
    % 精确率 = TP / (TP + FP)
    if (tp_per_class(class_idx) + fp_per_class(class_idx)) > 0
        precision_per_class(class_idx) = tp_per_class(class_idx) / (tp_per_class(class_idx) + fp_per_class(class_idx));
    else
        precision_per_class(class_idx) = 0;
    end
    
    % 召回率 = TP / (TP + FN)  
    if (tp_per_class(class_idx) + fn_per_class(class_idx)) > 0
        recall_per_class(class_idx) = tp_per_class(class_idx) / (tp_per_class(class_idx) + fn_per_class(class_idx));
    else
        recall_per_class(class_idx) = 0;
    end
    
    % F1 Score = 2 * (精确率 * 召回率) / (精确率 + 召回率)
    if (precision_per_class(class_idx) + recall_per_class(class_idx)) > 0
        f1_per_class(class_idx) = 2 * (precision_per_class(class_idx) * recall_per_class(class_idx)) / ...
                                 (precision_per_class(class_idx) + recall_per_class(class_idx));
    else
        f1_per_class(class_idx) = 0;
    end
end

% 宏平均F1 Score（所有类别F1的平均值）
f1_mean = mean(f1_per_class);

% % 显示结果
% fprintf('F1 Score计算结果:\n');
% fprintf('宏平均F1: %.4f\n', f1_mean);
% fprintf('各类别F1: ');
% fprintf('%.4f ', f1_per_class);
% fprintf('\n');
% 
% % 可选：显示详细统计信息
% fprintf('\n详细统计:\n');
% fprintf('类别\t精确率\t召回率\tF1-Score\tTP\tFP\tFN\n');
% for class_idx = 1:num_classes
%     fprintf('%d\t%.4f\t%.4f\t%.4f\t\t%d\t%d\t%d\n', ...
%             class_idx, precision_per_class(class_idx), recall_per_class(class_idx), ...
%             f1_per_class(class_idx), tp_per_class(class_idx), ...
%             fp_per_class(class_idx), fn_per_class(class_idx));
% end

end