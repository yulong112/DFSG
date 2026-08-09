function [AccRate] = fAccuracy(PredictLabel,TestLabel,Type)
% Computes classification accuracy.
% 列向量为一组标号
% Type -- 0,commonly used for accuracy under two-class or overall accracies
%          under multi-class.
%          -- 1, specilly used for individual accuracy under multi-class
%          -- 2, specilly used for individual average accuracy

if length(unique(TestLabel)) == 2
    PredictLabel(find(PredictLabel == 2)) = -1;
    AccRate = length(find((PredictLabel - TestLabel)==0))/length(TestLabel);
end

if  (length(unique(TestLabel))) > 2
    if Type == 1
            UniqueLabel = unique(TestLabel);
        for i = 1:length(UniqueLabel)
            TempTestLabel = TestLabel(find(TestLabel == UniqueLabel(i)));
            TempPredictLabel = PredictLabel(find(TestLabel == UniqueLabel(i)));
            AccRate(i,1) = length(find((TempPredictLabel - TempTestLabel)==0))/length(TempTestLabel);
        end
    end
    if Type == 2
            UniqueLabel = unique(TestLabel);
        for i = 1:length(UniqueLabel)
            TempTestLabel = TestLabel(find(TestLabel == UniqueLabel(i)));
            TempPredictLabel = PredictLabel(find(TestLabel == UniqueLabel(i)));
            AccRate(i,1) = length(find((TempPredictLabel - TempTestLabel)==0))/length(TempTestLabel);
        end
        AccRate = sum(AccRate)/length(UniqueLabel);
    end
    if Type == 0
        AccRate = length(find((PredictLabel - TestLabel)==0))/length(TestLabel);
    end
end
