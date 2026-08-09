function [ScaledTrainSample,MaxTemp1,MinTemp1] = ...
    fLibSVMTestNormalization(TestSampleAllDim,TrainMax,TrainMin)
% Sample X Dim
% HL

MaxTempMatrix1 = repmat(TrainMax,size(TestSampleAllDim,1),1);
MinTempMatrix1 = repmat(TrainMin,size(TestSampleAllDim,1),1);
ScaledTrainSample = (TestSampleAllDim - MinTempMatrix1)./(MaxTempMatrix1 - MinTempMatrix1);
