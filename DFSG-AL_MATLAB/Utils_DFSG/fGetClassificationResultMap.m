function [Map,TrainMap,TestMap] = fGetClassificationResultMap(TrainLabels, PredictLabels, TrainDataLoc, TestDataLoc, LabelMap)
% Computes classification accuracy.
% 列向量为一组标号
% Map:  TrainMap+TestMap
% TrainMap
% TestMap

UniLabel = unique(TrainLabels);
TrainMap =zeros(size(LabelMap));
TestMap =zeros(size(LabelMap));
for k = 1:length(UniLabel)
    IndexTrain = find(TrainLabels == UniLabel(k));
    IndexTest = find(PredictLabels == UniLabel(k));
    
    if size(IndexTrain,1)~=0
        [YCoordTrainSam, XCoordTrainSam] = f1DTo2DCoord(size(LabelMap),TrainDataLoc(IndexTrain));% coordinate of center point of mask
        TempTrainMap =zeros(size(LabelMap));
        for i=1:length(IndexTrain)
            TempTrainMap(YCoordTrainSam(i), XCoordTrainSam(i)) = UniLabel(k);
        end
        TrainMap = TrainMap+TempTrainMap;
    else
    end
    
    if size(IndexTest,1)~=0
        [YCoordTestSam, XCoordTestSam] = f1DTo2DCoord(size(LabelMap),TestDataLoc(IndexTest));
        TempTestMap =zeros(size(LabelMap));
        for i=1:length(IndexTest)
            TempTestMap(YCoordTestSam(i), XCoordTestSam(i)) = UniLabel(k);
        end
        TestMap = TestMap+TempTestMap;
    else
    end
end
Map=TrainMap+TestMap;