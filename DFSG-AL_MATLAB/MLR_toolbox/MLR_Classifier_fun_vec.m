function [maps,p]=MLR_Classifier_fun_vec(HyperVec,TrainSamLoc,TrainLabels,NormTestSam,lambda,sigma)

algorithm_parameters.lambda=0.001;
if exist('lambda','var')
    algorithm_parameters.lambda=lambda;
end
algorithm_parameters.beta = 0.5*algorithm_parameters.lambda;

if ~exist('sigma','var')
    sigma = 0.8;
end


    %% constuct img, train_with_labels, test_with_labels
    img = HyperVec';
    train=[TrainSamLoc,TrainLabels];
    train=train';
    %% class method : MLR
    %                 algorithm_parameters.mu = 4;
    trainset = img(:,train(1,:));
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
    
    learning_output = struct('scale',scale,'sigma',sigma);
    
    % learn the regressors
    [w,L] = LORSAL(K,train(2,:),algorithm_parameters.lambda,algorithm_parameters.beta);
    
    % compute the MLR probabilites
    p = mlr_probabilities(NormTestSam',trainset,w,learning_output);
    p = p';
    [~,maps] = max(p, [], 2);