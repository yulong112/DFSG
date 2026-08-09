function [PredLabels,PredLabels_ALL,confidence_coef] = SemiSupervised_Graph_Classifier_v2(Laplace_Mat, initial_Labels, TestSamLoc, alpha, TruthMap)
N=size(initial_Labels,1);
UU = (1-alpha) * speye(N) + alpha * Laplace_Mat;
F = UU \ initial_Labels; % classification function F = (I - \alpha S)^{-1}Y
[~, PredLabels] = max(F, [], 2); %simply checking which of elements is largest in each row
confidence_coef=F./repmat(sum(F,2),1,size(F,2));
PredLabels_ALL=PredLabels;
PredLabels=PredLabels(TestSamLoc);