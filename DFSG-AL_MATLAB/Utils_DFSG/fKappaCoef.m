function [kap,zscore] = fKappaCoef(PredictLabel,TestLabel)
% Computes classification accuracy.
% 列向量为一组标号
% Type - 0,commonly used for accuracy under two-class or overall accracies
%          under multi-class.
%      - 1, specilly used for individual accuracy under multi-class

%%
% kappa coefficient
% 
% confusion matrix :
%                            truth map
% predicted class     x      x

UniqueLabel = unique(TestLabel);
ConfMatrix = zeros(length(UniqueLabel),length(UniqueLabel));
for i = 1:length(UniqueLabel) % exhausitve calculation for all classes
    TempTestLabel = TestLabel(find(TestLabel == UniqueLabel(i)));% test labels belonging to a specific class
    TempPredictLabel = PredictLabel(find(TestLabel == UniqueLabel(i)));% predicted labels belonging to a specific class
    for j = 1:length(UniqueLabel)
        ConfMatrix(j,i) = length(find(TempPredictLabel == j));
    end
end

[kap,se,zscore,p0,SA]=kappa(ConfMatrix);

%%
function [kap,se,zscore,p0,SA]=kappa(H);
% kap	Cohen's kappa coefficient
%H: confusion matrix
% [kap,sd,z,OA,SA] = kappa(H);
%
% kap	Cohen's kappa coefficient point
% se	standard error of the kappa estimate
% H	data scheme (Concordance matrix or confusion matrix)
% z	z-score
% OA	overall agreement 
% SA	specific agreement 
%

N = sum(sum(H));

p0  = sum(diag(H))/N;  %accuracy of observed agreement, overall agreement 
%OA = sum(diag(H))/N);

p_i = sum(H); %sum(H,1);
pi_ = sum(H'); %sum(H,2)';

SA  = 2*diag(H)'./(p_i+pi_); % specific agreement 

pe  = (p_i*pi_')/(N*N);  % estimate of change agreement

px  = sum(p_i.*pi_.*(p_i+pi_))/(N*N*N);

%standard error 
kap = (p0-pe)/(1-pe);
sd  = sqrt((pe+pe*pe-px)/(N*(1-pe*pe)));

%standard error 
se  = sqrt((p0+pe*pe-px)/N)/(1-pe);
zscore = kap/se;
