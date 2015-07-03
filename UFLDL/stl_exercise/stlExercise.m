%% CS294A/CS294W Self-taught Learning Exercise

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  self-taught learning. You will need to complete code in feedForwardAutoencoder.m
%  You will also need to have implemented sparseAutoencoderCost.m and 
%  softmaxCost.m from previous exercises.
%   Step：
%   1，先对unlabeled data利用SAE（稀疏自编码）学习特征，得到input到L1的weight，bias
%   2，对labeled的train_data & test_data利用上面的weight，bias计算labeled的L1层的单元a；等于现在a就是输入了，和前面没关系了 
%   3，对L1层的单元，这里是200个，作为input，利用softmax有监督的学习L1-output的weight，之后用test_data和weight相乘进行预测
%   总结：觉得这里说是自学习，其实不是无监督的，和之前不同之处在于从input-L1层的weight，bias使用unlabeled
%   data基于SAE训练的，主要用人家unlabeled就是为了学习特征，其实就算用labeled一样可以学习到
%   并且因为0-9都是手写数字，利用5-9（unlabeled）学到的weight&bias应该对于0-4也能很好的适用，如果5-9和0-4特征差别很大，那
%   这套方法估计就行不通了。不过确实也是个特点，亮点在于利用unlabled学习特征。
%   再总结：1，input-L1：unlabled利用SAE学习特征，得到weight&bias
%           2，L1-output：先用labled通过weight&bias得到L1层单元，L1-output利用softmax分类（softmax可以理解为logistic，不要恐惧，这俩
%                           差不多）
%   补充：SAE为啥这么慢，人家好歹是三层网络，还需要反向传播，softmax仅仅需要向logistic一样计算各cost和grad一下就行了，简单算一下
%   如果SAE的L1层有200个单元，那么cost的计算就是softmax的200倍，人家还要反向传播，不算了，反正很多就是了
%   看到最新的UFLDL的stl教程是直接利用unlabeled进行学习特征，没有除去5-9，这样学到的特征对于分类应该更有效
%
%% ======================================================================
%  STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.

inputSize  = 28 * 28; % 输入层
numLabels  = 5;
hiddenSize = 200;       % 隐藏层
sparsityParam = 0.1; % desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		             %  in the lecture notes). 
lambda = 3e-3;       % weight decay parameter       
beta = 3;            % weight of sparsity penalty term   
maxIter = 400;      % 最大迭代次数，梯度下降多少次

%% ======================================================================
%  STEP 1: Load data from the MNIST database
%
%  This loads our training and test data from the MNIST database files.
%  We have sorted the data for you in this so that you will not have to
%  change it.

% Load MNIST database files
mnistData   = loadMNISTImages('softmax_exercise/mnist/train-images.idx3-ubyte');
mnistLabels = loadMNISTLabels('softmax_exercise/mnist/train-labels.idx1-ubyte');

% Set Unlabeled Set (All Images)

% Simulate a Labeled and Unlabeled set
labeledSet   = find(mnistLabels >= 0 & mnistLabels <= 4);
unlabeledSet = find(mnistLabels >= 5);

numTrain = round(numel(labeledSet)/2);
trainSet = labeledSet(1:numTrain);
testSet  = labeledSet(numTrain+1:end);

unlabeledData = mnistData(:, unlabeledSet(1:end/3));

trainData   = mnistData(:, trainSet);
trainLabels = mnistLabels(trainSet)' + 1; % Shift Labels to the Range 1-5

testData   = mnistData(:, testSet);
testLabels = mnistLabels(testSet)' + 1;   % Shift Labels to the Range 1-5

% Output Some Statistics
fprintf('# examples in unlabeled set: %d\n', size(unlabeledData, 2));
fprintf('# examples in supervised training set: %d\n\n', size(trainData, 2));
fprintf('# examples in supervised testing set: %d\n\n', size(testData, 2));

%% ======================================================================
%  STEP 2: Train the sparse autoencoder
%  This trains the sparse autoencoder on the unlabeled training
%  images. 

%  Randomly initialize the parameters
theta = initializeParameters(hiddenSize, inputSize);

%% ----------------- YOUR CODE HERE ----------------------
%  Find opttheta by running the sparse autoencoder on
%  unlabeledTrainingImages
% 先用SAE学习特征，input到L1层
opttheta = theta; 
addpath minFunc/
options.Method = 'lbfgs';
options.maxIter = 50;
options.display = 'on';
[opttheta, loss] = minFunc( @(p) sparseAutoencoderCost(p, ...
      inputSize, hiddenSize, ...
      lambda, sparsityParam, ...
      beta, unlabeledData), ...
      theta, options);

%% -----------------------------------------------------
                          
% Visualize weights
W1 = reshape(opttheta(1:hiddenSize * inputSize), hiddenSize, inputSize);
display_network(W1');

%%======================================================================
%% STEP 3: Extract Features from the Supervised Dataset
%  
%  You need to complete the code in feedForwardAutoencoder.m so that the 
%  following command will extract features from the data.
% 前向传播，还专门写个函数，其实就是: a = sigmoid(W1*data+repmat(b1,[1,size(data,2)]));
trainFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       trainData);

testFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       testData);

%%======================================================================
%% STEP 4: Train the softmax classifier

softmaxModel = struct;  
%% ----------------- YOUR CODE HERE ----------------------
%  Use softmaxTrain.m from the previous exercise to train a multi-class
%  classifier. 

%  Use lambda = 1e-4 for the weight regularization for softmax

% You need to compute softmaxModel using softmaxTrain on trainFeatures and
% trainLabels
% 输入层变为L1层，output变成分类的结果，这里是0~4，有labels，有监督的
lambda = 1e-4;
inputSize = hiddenSize;
numClasses = numel(unique(trainLabels));%unique为找出向量中的非重复元素并进行排序

%% -----------------------------------------------------
options.maxIter = 100;
softmaxModel = softmaxTrain(inputSize, numClasses, lambda, ...
                            trainFeatures, trainLabels, options);

%%======================================================================
%% STEP 5: Testing 

%% ----------------- YOUR CODE HERE ----------------------
% Compute Predictions on the test set (testFeatures) using softmaxPredict
% and softmaxModel
[pred] = softmaxPredict(softmaxModel, testFeatures);

%% -----------------------------------------------------

% Classification Score
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));

% (note that we shift the labels by 1, so that digit 0 now corresponds to
%  label 1)
%
% Accuracy is the proportion of correctly classified images
% The results for our implementation was:
%
% Accuracy: 98.3%
%
% 
