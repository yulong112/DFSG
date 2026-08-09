function [ScaledTrainSample,MaxTemp1,MinTemp1] = fLibSVMTrainNormalization(TrainSampleAllDim)
% Sample X Dim
% HL

MaxTemp1 = max(TrainSampleAllDim);
MinTemp1 = min(TrainSampleAllDim);
MaxTempMatrix1 = repmat(MaxTemp1,size(TrainSampleAllDim,1),1);
MinTempMatrix1 = repmat(MinTemp1,size(TrainSampleAllDim,1),1);
ScaledTrainSample = (TrainSampleAllDim - MinTempMatrix1)./(MaxTempMatrix1 - MinTempMatrix1);
