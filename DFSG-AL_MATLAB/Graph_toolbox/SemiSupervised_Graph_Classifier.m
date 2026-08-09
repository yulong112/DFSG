function PredLabels = SemiSupervised_Graph_Classifier(S_W, initial_Labels, TestSamLoc, alpha)
N=size(initial_Labels,1);
UU = speye(N) - alpha * S_W;
F = UU \ initial_Labels; % classification function F = (I - \alpha S)^{-1}Y
[~, PredLabels] = max(F, [], 2); %simply checking which of elements is largest in each row

PredLabels=PredLabels(TestSamLoc);