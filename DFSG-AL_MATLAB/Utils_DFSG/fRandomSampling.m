function [Subset1,Subset2,Subset1Loc,Subset2Loc] = fRandomSampling(Samples,SamplingNum,SeedNo)
%
rand('seed',SeedNo);
UniformRandData1 = rand(size(Samples,1),1);
[SortedIndex1, Index1] = sort(UniformRandData1,'descend');%for random sampling
Subset1 = Samples(Index1(1:SamplingNum),:);
Subset2 = Samples(Index1(SamplingNum+1:end),:);
Subset1Loc=Index1(1:SamplingNum);
Subset2Loc=Index1(SamplingNum+1:end);