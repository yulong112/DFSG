function [PredLabels,TotalReconErr,TotalReconEnergy] = fSRClassifierTestV2(TrainSet, TestSet, TrainLabels, Weight, KerPara)
%TrainSet, TestSet: SamNum X Dim
%Syntax: require atoms of a class are aggregated as a submatrix in dictionary
%
% Description
%   	Get predicted label of test set via its sparse solution
% Inputs
% 	TrainSet -- Sample Number X Dimension
%   TestSet -- Sample Number X Dimension
%   TrainLabels -- Sample Number X 1
%    Weight -- sparse solution corresponding to test set
% Outputs
%    PredLabels -- predicted label
%    TotalErrs -- reconstruction error.
%                      sample number X reconstruction errors in all the classes
% See Also
%
% References
% 	[1]
% -------------------------------------------------------------------------
% Copyrights
% 	Contributor:
% 	CodeAuthor:

%% re-arrage the sample in order to make samples of the same class in aggregation
Labels = unique(TrainLabels);
Dict = TrainSet';

TotalReconErr = [];
for LabelNo = 1:length(Labels)
    TempTrainSet = TrainSet(find(TrainLabels==Labels(LabelNo)),:);
    TempWeight = Weight(find(TrainLabels==Labels(LabelNo)),:);
    Reconstruct = TempTrainSet'*TempWeight;
    %ReconErr = (TestSet'-Reconstruct);
    ReconErr = sum((TestSet'-Reconstruct).*(TestSet'-Reconstruct));
    TotalReconErr = [TotalReconErr; ReconErr];
end
[err, PredLabels] = min(TotalReconErr);
PredLabels = PredLabels';
% ClassNo = unique(TrainLabels);
% ClassNum = length(ClassNo);
% for ClassNoNo = 1:ClassNum
%     Ns(ClassNoNo) = length(find(TrainLabels==ClassNo(ClassNoNo)));% the number of train samples in a certain class
% end
% SamNum = size(TrainSet,1);
% Sign = 0;
% 
% if SamNum>length(TrainLabels)
%     Sign = 1;
% end
% %%
% if iscell(Weight)==0
%     D = TrainSet';
%     TotalErrs =[];
%     TotalReconEnergy = [];
%     for TestSamNo = 1:size(Weight,2)
%         y = TestSet(TestSamNo,:)';
%         xp = Weight(:,TestSamNo);
%         errs = zeros(1, ClassNum);
%         t = 1;
%         for k = 1 : ClassNum
%             cols = t : t+Ns(k)-1;%
%             if Sign==0
%                 errs(k) = norm(y - D(:, cols) * xp(cols));
%                 ReconEnergy(k) = norm(D(:, cols) * xp(cols));
%             end
%             if Sign==1 % 用于带常数项的情况
%                 Constcols = length(TrainLabels)+1: SamNum;%
%                 ConstErrs =  D(:, Constcols) * xp(Constcols);
%                 errs(k) = norm(y - D(:, cols) * xp(cols)-ConstErrs);
%             end
%             t = t + Ns(k);
%         end
%         [e, id] = min(errs);
%         PredLabels(TestSamNo,1) = id;
%         TotalErrs =[TotalErrs;errs];
%         TotalReconEnergy =[TotalReconEnergy;ReconEnergy];
%     end
% end
% 
% if nargin ==4
%     if iscell(Weight)==1
%         D = TrainSet';
%         TotalErrs =[];
%         for TestSamNo = 1:size(Weight,2)
%             y = TestSet{TestSamNo}';
%             %        xp = Weight(:,TestSamNo);
%             xp = Weight{TestSamNo};
%             errs = zeros(1, ClassNum);
%             t = 1;
%             for k = 1 : ClassNum
%                 cols = t : t+Ns(k)-1;%
%                 errs(k) = norm(y - D(:, cols) * xp(cols,:),'fro');
%                 t = t + Ns(k);
%             end
%             [e, id] = min(errs);
%             PredLabels(TestSamNo,1) = id;
%             TotalErrs =[TotalErrs;errs];
%         end
%     end
% end
% 
% if nargin ==5
%     D = TrainSet';
%     TotalErrs =[];
%     for TestSamNo = 1:size(Weight,2)
%         y = TestSet{TestSamNo}';
%         yKer = diag(fKernelGen(y', y', KerPara.KerType, KerPara.Gamma));
%         xp = Weight{TestSamNo};
%         errs = zeros(1, ClassNum);
%         t = 1;
%         for k = 1 : ClassNum
%             cols = t : t+Ns(k)-1;%
%             k1 = diag(2*xp(cols,:)'*fKernelGen(D(:, cols)', y',KerPara.KerType, KerPara.Gamma));
%             k2 = diag(xp(cols,:)'*fKernelGen(D(:, cols)', D(:, cols)',KerPara.KerType, KerPara.Gamma) * xp(cols,:));
%             errs(k) = norm(yKer - k1 + k2);
%             t = t + Ns(k);
%         end
%         [e, id] = min(errs);
%         PredLabels(TestSamNo,1) = id;
%         TotalErrs =[TotalErrs;errs];
%     end
% end

