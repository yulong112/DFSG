function [Weight STDLevel] = fSRClassifierTrain(TrainSet, TestSet, TrainLabels, Method,maxIters, lambda, OptTol)
% TrainSet, TestSet: SamNum X Dim
% Syntax
%
% Description
% 	Sparse Representation Classifier by l1 constrain
% Inputs
% 	TrainSet -- sample number X dimension
%   TestSet -- sample number X dimension
%   TrainSet -- sample number X 1
%   method  -- determine the l1 solver
%                  StandardLS - standard or ordinary or classical least square
%   Ns -- Number of training class
% Outputs
%   Weight: sparse solution corresponding to test set.
% See Also
% TrainLabels- is useless for BP
% References
% 	[1]
% -------------------------------------------------------------------------
% Copyrights
% 	Contributor:
% 	CodeAuthor:

%% re-arrage the sample in order to make samples of the same class in aggregation  
% ClassNo = unique(TrainLabels);
% ClassNum = length(ClassNo);
% TempTrainSet=[];
% TempTrainLabels = [];
% for ClassNoNo = 1:ClassNum
%     Ns(ClassNoNo) = length(find(TrainLabels==ClassNo(ClassNoNo)));
%     % Ns -- Number of samples in each training class
%     TempTrainSet = [TempTrainSet;TrainSet(find(TrainLabels==ClassNo(ClassNoNo)),:)];
%     TempTrainLabels = [TempTrainLabels;TrainLabels(find(TrainLabels==ClassNo(ClassNoNo)),:)];
% end
% %%%%%%%%%%
% TrainSet = TempTrainSet;
% TrainLabels = TempTrainLabels;

%% choose the sparse representation method
global Counter % for counting time
global TimePer  

if strcmpi(Method,'StandardLS')||strcmpi(Method,'RidgeRegrssion_MatToolbox')
    D = TrainSet';
    ResponseSet = TestSet';
    Weight = [];
    for i = 1:size(ResponseSet,2)
        if strcmpi(Method,'StandardLS')
        TempWeight = D\ResponseSet(:,i);
        end
        if strcmpi(Method,'RidgeRegrssion_MatToolbox')
        TempWeight = ridge(ResponseSet(:,i), D,maxIters,1);% 1: without intercept. and maxIters as the set of regularization parameters
        end
%         if strcmpi(Method,'RidgeRegrssion_HighEfficience')
%             IdentityMatrix = eye(size(D,2));
%             TempMatrix = (D'*D+maxIters*IdentityMatrix)*D';
%         end
        Weight = [Weight, TempWeight];
    end
end

if strcmpi(Method,'RidgeRegrssion_HighEfficience')||strcmpi(Method,'StdLS_HighEfficience')
    D = TrainSet';
    ResponseSet = TestSet';
    Weight = [];
        if strcmpi(Method,'RidgeRegrssion_HighEfficience')
            IdentityMatrix = eye(size(D,2));
            TempMatrix = (D'*D+maxIters*IdentityMatrix)^(-1)*D';
            Weight = TempMatrix*ResponseSet;% coefficients connected to test samples
            Weight_TrainSam = TempMatrix*D;
            Residual = [D,ResponseSet] - D*[Weight_TrainSam,Weight];
            STDLevel = std(Residual')'; % standard variance level of each of all the bands
        end
        if strcmpi(Method,'StdLS_HighEfficience')
            TempMatrix = (D'*D)^(-1)*D';
            Weight = TempMatrix*ResponseSet;% coefficients connected to test samples
            Weight_TrainSam = TempMatrix*D;
            Residual = [D,ResponseSet] - D*[Weight_TrainSam,Weight];
            STDLevel = std(Residual')'; % standard variance level of each of all the bands
        end
end

if strcmpi(Method,'KernelRidgeRegrssion_HighEfficience')
    D = TrainSet';
    ResponseSet = TestSet';
    Weight = [];
        if strcmpi(Method,'KernelRidgeRegrssion_HighEfficience')
            IdentityMatrix = eye(size(D,2));
            TempMatrix1 = (fKernelGen(D', D', maxIters.KerType, maxIters.Gamma) + maxIters.RegularizationPara*IdentityMatrix)^(-1);
            %TempMatrix2 = fKernelGen(TempMatrix1,D, maxIters.KerType, maxIters.Gamma);
            TempMatrix2 = fKernelGen(D', ResponseSet', maxIters.KerType, maxIters.Gamma);
            Weight = TempMatrix1*TempMatrix2;
            %TempMatrix = (D'*D+maxIters*IdentityMatrix)^(-1)*D';
            %Weight = TempMatrix*ResponseSet;% coefficients connected to test samples
            %Weight_TrainSam = TempMatrix*D;
            %Residual = [D,ResponseSet] - D*[Weight_TrainSam,Weight];
            %STDLevel = std(Residual')'; % standard variance level of each of all the bands
        end
end

if strcmpi(Method,'l1qc_logbarrier')||strcmpi(Method,'l1eq_pd')||strcmpi(Method,'pfp')...
        ||strcmpi(Method,'BasicPursuit')||strcmpi(Method,'Homotopy')||strcmpi(Method,'BP')||...
        strcmpi(Method,'OMP')||strcmpi(Method,'glmnet')
    D = TrainSet';
    [Dim, AtomNum] = size(D);
    Weight = [];
    for TestSamNo = 1:size(TestSet,1)
        y = TestSet(TestSamNo,:)';
        x0 = D'*y;% initilization
        %
        if strcmpi(Method,'glmnet')
%             fit1=glmnet(D,y);
%             [PredLabels, errs] = fSRClassifierTest(NormTrainSam, NormTestSam, TrainLabels, Weight);
%             AccRate(i,j) = fAccuracy(PredLabels,TestLabels,0);% 每个选择的尺度上的测试样本准确率
        end
        if strcmpi(Method,'l1qc_logbarrier')
            xp = l1qc_logbarrier(x0, D, [], y, 1e-3, 1e-3);
        end
        if strcmpi(Method,'l1eq_pd')
            xp = l1eq_pd(x0, D, [], y);
        end
        if strcmpi(Method,'pfp')
            %xp = bp.pfp(y, D);
            TH.max_iter = 200;  TH.min_tol = 1e-2;
            xp = gbp(y, D,TH);
        end
        if strcmpi(Method,'BasicPursuit')
            xp = fBasicPursuit(D, y,'linprog');
        end
        if strcmpi(Method,'Homotopy')
            [xp, iters, activeSet] = SolveLasso(D, y);
        end
        if strcmpi(Method,'BP')  % sparse lab, solving basis pursuit
            xp = SolveBP(D, y, AtomNum,maxIters, lambda, OptTol);
        end
        if strcmpi(Method,'LARSLasso')  % sparse lab, solving lasso with LARS 
            xp = SolveBP(D, y, AtomNum,maxIters, lambda, OptTol);
        end
        if strcmpi(Method,'OMP')
            xp = SolveOMP(D, y, AtomNum, maxIters, 1e-7, 0, 0, OptTol);
        end
        %     if strcmpi(Method,'SOMP')
        %     xp = SolveSOMP(D, y, AtomNum, maxIters, 1e-7, 0, 0, OptTol);
        %     end
        % when we conduct basic pursuit on samples, it may be not convergent (interior point method)!
        if xp==0
            TestSamNo
            xp = zeros(size(D,2),1);
            PredLabels(TestSamNo,1) = -1;
            Weight = [Weight,xp];
            continue;
        end
        Weight = [Weight,xp];
    end
end
%% specialized for SOMP
if strcmpi(Method,'SOMP')
    D = TrainSet';
    clear TrainSet
    [Dim, AtomNum] = size(D);
    for TestSamNo = 1:length(TestSet)
        %%%%%%%%%%%
        y = TestSet{TestSamNo}';
        %xp = SolveSOMP(D, y, AtomNum, maxIters, 1e-7, 0, 0, OptTol);
        %Weight{TestSamNo} = xp;% 原有
         %%%%%%% 
        %
        %xp = SolveSOMP(D, y, AtomNum, maxIters, 1e-7, 0, 0, OptTol);
        [xp indSet Approx Res] = fSimulOMP2(y, D, 1e-7, maxIters, 3);% function writen by Chen Yi
        xp1{1} = xp;
        [PredLabels(TestSamNo,1), errs] = fSRClassifierTest(D', TestSet(TestSamNo), TrainLabels, xp1);
        Weight = PredLabels;
    end
end

if strcmpi(Method,'JointRidgeRegression')
    D = TrainSet';
    clear TrainSet
    [Dim, AtomNum] = size(D);
    for TestSamNo = 1:length(TestSet)
        %%%%%%%%%%%
        y = TestSet{TestSamNo}';
        xp=(D'*D+lambda*eye(size(D,2)))^(-1)*D'*y;
        xp1{1} = xp;
        [PredLabels(TestSamNo,1), errs] = fSRClassifierTest(D', TestSet(TestSamNo), TrainLabels, xp1);
        %Weight = PredLabels;
    end
    Weight = PredLabels;
end

%% Improved SOMP
if strcmpi(Method,'ImprovedSOMP')
    NumOfNeigh = size(TrainSet{1},1);
    D = [];
    for i = 1:length(TrainSet)
        D = [D;TrainSet{i}];
    end
    D = D';
    clear TrainSet
    %[Dim, AtomNum] = size(D);
    for TestSamNo = 1:length(TestSet)
        %%%%%%%%%%%
        y = TestSet{TestSamNo}';
        [xp TempD indSet Approx Res] = fImprovedSimulOMP1(y, D, NumOfNeigh,1e-7, maxIters, 3);% function writen by Chen Yi
        xp1{1} = xp;
        [PredLabels(TestSamNo,1), errs] = fSRClassifierTest(TempD', TestSet(TestSamNo), TrainLabels, xp1);
        Weight = PredLabels;
    end
end

%% specialized for SSP
if strcmpi(Method,'SSP')
    D = TrainSet';
    clear TrainSet
    [Dim, AtomNum] = size(D);
    for TestSamNo = 1:length(TestSet)
        %%%%%%%%%%%
        y = TestSet{TestSamNo}';
        [xp indSet Approx Res] = fSimulSP(y, D, 1e-7, maxIters, 3);% function writen by Chen Yi
        xp1{1} = xp;
        [PredLabels(TestSamNo,1), errs] = fSRClassifierTest(D', TestSet(TestSamNo), TrainLabels, xp1);
        Weight = PredLabels;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
%     errs = zeros(1, ClassNum);
%     t = 1;
%     %try
%     for k = 1 : ClassNum
%         cols = t : t+Ns(k)-1;%
%         errs(k) = norm(y - D(:, cols) * xp(cols));
%         t = t + Ns(k);     
%     end
%     
%     [e, id] = min(errs);
%     PredLabels(TestSamNo,1) = id;
%    
% end
