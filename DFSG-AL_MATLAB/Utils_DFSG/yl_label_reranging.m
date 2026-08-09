function TruthMap=yl_label_reranging(TruthMap)

    TruthMap1D = TruthMap(:);
    UniqueLabel = unique(TruthMap1D);
    UniqueLabel = sort(UniqueLabel, 'ascend');
    for i = 1: length(UniqueLabel)
        TruthMap1D( find(TruthMap1D==UniqueLabel(i)) ) = i-1;
    end
    TruthMap = reshape(TruthMap1D, size(TruthMap));