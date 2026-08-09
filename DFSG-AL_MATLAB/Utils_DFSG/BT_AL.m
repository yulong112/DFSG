function [varargout] = BT_AL(varargin)
%
%
% LORSAL_MLL_AS: Hyperspectral segmentation
%
% [class_results,seg_results]=LORSAL_MLL_AS(image,train,test,learning_method,...
%                           algorithm_parameters,AL_sampling);
%
% AL_sampling:
%             candidate: candidate training set
%             U:
%             u: new samples per iteration
%
% Brief description
%
% This demo implements the algorithm introduced in [1] (see also [2,3]).
%
% In summary:
%
%   1- Based on a training set containing  spectral vectors and the
%      respective labels, learn the regressors of a multinomial
%      logistic regression (MLR) using LORSAL (see appendix of [1] and [3])
%
%    2- Based on a multi-level logistic (MLL) prior, to model the contextual
%       spatial information of the hyperspectral images, compute the MAP
%       segmentation  via the \alpha-Expanasion Graph-cut algorithm.
%
%    3- Based on the mutual information between the MLR regressors and the
%       class labels, acively select new  samples.
%
%    4- Goto 1, until some stopping rule is meet
%
%
%
% -------------- Input parameters -----------------------------------------
%
% img     -  hyperspectral dataset:  3 \times 1 struct
%            im:   no_bands \times no_pixels
%            size :  no_lines \time no_columns
%            mycolormap : my colormap
%
% train   -  training set  :    2 \times no_train
%            the first line is the indexes
%            the second line is the labeles
%
% test    -  test set  :  2 \times n_test
%            the first line is the indexes
%            the second line is the labeles
%             default = train;
%
% learning_method  -  this variable takes values in  {linear, RBF}
%                     default = RBF.
%
% GC_toolbox        - this variable takes values in  {Bagon, Bioucas}
%                     default = Bagon.
%                     Graph cut toolbox to implement the alpha-Expansion
%                     algorithm. Both toolboxes are compiled for
%                     Matlab R2009.
%                     If you face problems, compile the Bagon toolbox in
%                     you computer by running compile_gc.m. See details in
%                     GCmex1.2
%
% algorithm_parameters: 3 \times 1 struct
%
% lambda - the Laplace parameter controlling the degree of sparsity of the
%          regressor components.
%
% beta   -  LORSAL parameter setting the augmented Lagrance weight  and
%           algorithm convergency speed (see appendix of [1]; reasonable
%           values: beta < lambda)
% mu      - the spatial prior regularization  parameter, controlling
%           the degree of spatial smothness. Tune this parameter to
%           obtain a good segmentation results (reasobnable values [1, 4]).
%
% AL_sampling:   4 \times 1 struct
%
% AL_method -  this variable takes values in  {RS, MI, BT, MBT}
%              default = RS.
%               RS  --  random selection
%               MI  --  maximum entropy
%               BT  --  breaking ties
%              MBT  --  mordified breaking ties
%
% candidate - candidate training set
% U         - the size of actively selected samples
% u         -  new samples per iteration
%
%
%
% --- output parameters ---------------------------------------------------
%
% class_results:   classification results, just LORSAL-AL, without spatial
%          information
%   ------ 4 \times 1 struct,
%
% map     -   classification map
% OA      -   classification overall accuracy
% AA      -   classification average accuracy
% kappa   -   classification kappa statistic
% CA      -   classification class individual accuracy

% seg_results:   segmentation results, just LORSAL-AL-MLL, with spatial
%          information
%   ------ 4 \times 1 struct,
%
% map     -   segmentation map
% OA      -   segmentation overall accuracy
% AA      -   segmentation average accuracy
% kappa   -   segmentation kappa statistic
% CA      -   classification class individual accuracy
%
%
% -------------------------------------------------------------------------
%
% More details in
%
% [1] J. Li, J. Bioucas-Dias, and A. Plaza, "Hyperspectral segmentation
%  with active learning," submitted to IEEE TGRS,  2010 (link).
%
%  [2] J. Li, J. Bioucas-Dias, and A. Plaza, "Semi-supervised Hyperspectral
%  classification using active label selection," in SPIE Europe Remote
%  Sensing, Vol.7477, 2009.
%
%  [3] J. Bioucas-Dias and M. Figueiredo,  "Logistic regression via variable
%  splitting and augmented lagrangian tools"  Tech. Rep., Insituto Superior
%  Tecnico, TULisbon, 2009
%
%
%  Copyright: Jun Li (jun@lx.it.pt)
%             &
%             Jos?Bioucas-Dias (bioucas@lx.it.pt)
%
%  For any comments contact the authors

% if nargout > 2, error('too many output parameters'); end

img = varargin{1}; % 1st parameter is the data set
if ~numel(img),error('the data set is empty');end


train = varargin{2}; % the 2nd parameter is the training set
if isempty(train), error('the train data set is empty, please provide the training samples');end
no_classes = max(train(2,:));

if nargin >2
    test = varargin{3}; % the 3rd parameter is the test
else
    fprintf('the test set is empty, thus we use the training set as the validation set \n')
    test = train;
end

if nargin >3
    learning_method = varargin{4};
else
    learning_method = 'RBF';
end

if isempty(learning_method)    learning_method = 'RBF';end

if nargin >4
    algorithm_parameters = varargin{5};
else
    algorithm_parameters.lambda=0.001;
    algorithm_parameters.beta = 0.5*algorithm_parameters.lambda;
    algorithm_parameters.mu = 4;
end

if isempty(algorithm_parameters)
    algorithm_parameters.lambda=0.001;
    algorithm_parameters.beta = 0.5*algorithm_parameters.lambda;
    algorithm_parameters.mu = 4;
end

if nargin >5
    AL_sampling = varargin{6};
else
    AL_sampling = [];
end

if isempty(AL_sampling)
    tot_sim = 1;
else
    tot_sim = AL_sampling.U/AL_sampling.u ;
end

% if nargin >6
%     class_method = varargin{7};
% else
%     class_method = 'LS';
% end
% 
% if isempty(class_method)
%     class_method = 'LS';
% end

if nargin >6
    flg = varargin{7};
else
    flg=0;
end
AccRate=[];

%         maxIters = 300;
%         OptTol= 0.005;
% start active section iterations
for iter = 1:tot_sim
    %     fprintf('Active selection iteration  %d \n', iter);
    
    if flg == 1;
        test = AL_sampling.candidate;
    end
    
    %% 以下为原论文中MLR法
    trainset = img.im(:,train(1,:));
        if strcmp(learning_method,'RBF')
            sigma = 0.8;
            % build |x_i-x_j| matrix
            nx = sum(trainset.^2);
            [X,Y] = meshgrid(nx);
            dist=X+Y-2*trainset'*trainset;
            clear X Y
            scale = mean(dist(:));
            % build design matrix (kernel)
            K=exp(-dist/2/scale/sigma^2);
            clear dist
            % set first line to one
            K = [ones(1,size(trainset,2)); K];
        else
            K =[ones(1,size(trainset,2)); trainset];
            scale = 0;
            sigma = 0;
        end
    
        learning_output = struct('scale',scale,'sigma',sigma);
    
        % learn the regressors
        [w,L] = LORSAL(K,train(2,:),algorithm_parameters.lambda,algorithm_parameters.beta);
    
        % compute the MLR probabilites
        p = mlr_probabilities(img.im,trainset,w,learning_output);
        [~,maps] = max(p);
        PredLabels=maps(test(1,:))';
        AccRate(iter) = fAccuracy(PredLabels,test(2,:)',0);% 每个选择的尺度上的测试样本准确率
    %%%%% 以上为MLR法
    %%  LS和BP法：compute the classification results 和分属各类的误差
%     PredLabels=[];
%     errs=[];
%     pactive=[];
%     testset=img.im(:,AL_sampling.candidate(1,:));
%     
%         trainset = fNormDic(trainset')';
%         testset = fNormDic(testset')';
%     if strcmp(class_method,'LS')
%         [Weight] = ...
%             fSRClassifierTrain(trainset', testset', train(2,:)', 'StdLS_HighEfficience');
%         %[TempPredLabels, errs] = fSRClassifierTest(NormTrainSam, TempTestSam, TrainLabels, Weight);
%         [PredLabels, errs] = fSRClassifierTestV2(trainset', testset', train(2,:)', Weight);
%     else strcmp(class_method,'BP')
%         
%         [Weight] = ...
%             fSRClassifierTrain(trainset', testset', train(2,:)', 'BP',maxIters,0,OptTol);
%        [PredLabels, errs] = fSRClassifierTest(trainset', testset', train(2,:)', Weight);
%         
%     end
%     errsum=sum(errs,1);
%     errs=errs./repmat(errsum,size(errs,1),1);
%     pactive=1-errs;
%     pactive=pactive./repmat(sum(pactive,1),size(pactive,1),1);
    %     [maxp, class_results.map] = max(p);
    %     [class_results.OA(iter),class_results.kappa(iter),class_results.AA(iter),...
    %     class_results.CA(iter,:)]= calcError(test(2,:)-1, class_results.map(test(1,:))-1,[1:no_classes]);
    %
    
    if isempty(AL_sampling)
        fprintf( 'No active selection sampling will be addressed \n' );
        %     elseif strcmp(AL_sampling.AL_method,'RS');
        %         per_indexR = randperm(size(AL_sampling.candidate,2));
        %         xp = per_indexR(1:AL_sampling.u);
        %         trainnew = AL_sampling.candidate(:,xp);
        %         train = [train,trainnew];
        %         AL_sampling.candidate(:,xp) = [];
        %
        %     elseif strcmp(AL_sampling.AL_method,'MI');
        %         pactive = p(:,AL_sampling.candidate(1,:));
        %         pactive = sort(pactive,'descend');
        %         pactive_minME = pactive(1,:)-pactive(no_classes,:);
        %         [sortmminppME, indexsortminppME] = sort(pactive_minME);
        %         xp = indexsortminppME(1:AL_sampling.u);
        %         trainnew = AL_sampling.candidate(:,xp);
        %         train = [train,trainnew];
        %         AL_sampling.candidate(:,xp) = [];
        
    elseif strcmp(AL_sampling.AL_method,'BT');
        pactive = p(:,AL_sampling.candidate(1,:));
        pactive = sort(pactive,'descend');
        pactive_minBT = pactive(1,:)-pactive(2,:);
        [sortmminppBT, indexsortminppBT] = sort(pactive_minBT);
%         xp = indexsortminppBT(1:1);
        xp = indexsortminppBT(1:AL_sampling.u);
        trainnew = AL_sampling.candidate(:,xp);
        train = [train,trainnew];
        AL_sampling.candidate(:,xp) = [];
        
        %     else strcmp(AL_sampling.AL_method,'MBT');
        %         pactive = p(:,AL_sampling.candidate(1,:));
        %         [maxp, class] = max(pactive);
        %              xp = [];
        %              xpp = [];
        %              for class_k = 1:no_classes
        %                  classk_estMC = find(class ==class_k);
        %                  pclassk_estMC = pactive(:,classk_estMC);
        %                  pclassk_estMC(class_k,:) = [];
        %                  pclassk_maxMC = max(pclassk_estMC);
        %                  if length(classk_estMC)>(ceil(AL_sampling.u/no_classes))
        %                      [pclassk_nextMC,pclassk_nextindexMC] = sort(pclassk_maxMC,'descend');
        %                      xpkindexMC = pclassk_nextindexMC(1:ceil(AL_sampling.u/no_classes));
        %                      xp = [xp classk_estMC(xpkindexMC)];
        %                      xpp = [xpp pclassk_nextMC(1:ceil(AL_sampling.u/no_classes))];
        %                  else
        %                      xp = [xp classk_estMC];
        %                      xpp = [xpp pclassk_maxMC];
        %                  end
        %              end
        %
        %              [xpp_p,xpp_index] = sort(xpp,'descend');
        %              xp = xp(xpp_index(1:end));
        %              trainnew = AL_sampling.candidate(:,xp);
        %              train = [train,trainnew];
        %              AL_sampling.candidate(:,xp) = [];
    end
end

if tot_sim>0
    iter_num=0:tot_sim-1;
    figure
    plot(iter_num,AccRate,'-bo','LineWidth',2,...
        'MarkerEdgeColor','k',...
        'MarkerFaceColor','r',...
        'MarkerSize',8);
    % set(handles,'ytick',0:0.1:1) % handles可以指定具体坐标轴的句柄
    axis([0 tot_sim 0 1]);
    % title('测试样本预测准确率随迭代次数的变化-含高斯预处理，每类训练样本1个')
    xlabel('Number of Active Set Selections')
    ylabel('Accuracy')
    set(gca,'FontSize',20);
    set(get(gca,'YLabel'),'Fontsize',20);
    set(get(gca,'XLabel'),'Fontsize',20);
end

%% class method : LS分类器
% trainset=img.im(:,train(1,:));
% testset=img.im(:,test(1,:));
% trainlabel=train(2,:);
% testlabel=test(2,:);
%         trainset = fNormDic(trainset')';
%         testset = fNormDic(testset')';
% 
%         if strcmp(class_method,'LS')
%             [Weight] = ...
%                 fSRClassifierTrain(trainset', testset', trainlabel', 'StdLS_HighEfficience');
%             %[TempPredLabels, errs] = fSRClassifierTest(NormTrainSam, TempTestSam, TrainLabels, Weight);
%             [PredLabels, errs] = fSRClassifierTestV2(trainset', testset', trainlabel', Weight);
%         else strcmp(class_method,'BP')
%             [Weight] = ...
%                 fSRClassifierTrain(trainset', testset', trainlabel', 'BP',maxIters,0,OptTol);
%             [PredLabels, errs] = fSRClassifierTest(trainset', testset', trainlabel', Weight);
%         end
%             AccRate = fAccuracy(PredLabels,testlabel',0)% 每个选择的尺度上的测试样本准确率

test=AL_sampling.candidate;
% varargout(1) ={AccRate};
% if nargout == 2, varargout(2) = {PredLabels};end
varargout(1) ={train};
varargout(2) ={AccRate};
varargout(3) ={test};

return

% %-----------------------------------------------------------------------%