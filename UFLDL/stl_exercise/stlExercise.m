%% CS294A/CS294W Self-taught Learning Exercise

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  self-taught learning. You will need to complete code in feedForwardAutoencoder.m
%  You will also need to have implemented sparseAutoencoderCost.m and 
%  softmaxCost.m from previous exercises.
%   Step��
%   1���ȶ�unlabeled data����SAE��ϡ���Ա��룩ѧϰ�������õ�input��L1��weight��bias
%   2����labeled��train_data & test_data���������weight��bias����labeled��L1��ĵ�Ԫa����������a���������ˣ���ǰ��û��ϵ�� 
%   3����L1��ĵ�Ԫ��������200������Ϊinput������softmax�мල��ѧϰL1-output��weight��֮����test_data��weight��˽���Ԥ��
%   �ܽ᣺��������˵����ѧϰ����ʵ�����޼ල�ģ���֮ǰ��֮ͬ�����ڴ�input-L1���weight��biasʹ��unlabeled
%   data����SAEѵ���ģ���Ҫ���˼�unlabeled����Ϊ��ѧϰ��������ʵ������labeledһ������ѧϰ��
%   ������Ϊ0-9������д���֣�����5-9��unlabeled��ѧ����weight&biasӦ�ö���0-4Ҳ�ܺܺõ����ã����5-9��0-4�������ܴ���
%   ���׷������ƾ��в�ͨ�ˡ�����ȷʵҲ�Ǹ��ص㣬������������unlabledѧϰ������
%   ���ܽ᣺1��input-L1��unlabled����SAEѧϰ�������õ�weight&bias
%           2��L1-output������labledͨ��weight&bias�õ�L1�㵥Ԫ��L1-output����softmax���ࣨsoftmax�������Ϊlogistic����Ҫ�־壬����
%                           ��ࣩ
%   ���䣺SAEΪɶ��ô�����˼Һô����������磬����Ҫ���򴫲���softmax������Ҫ��logisticһ�������cost��gradһ�¾����ˣ�����һ��
%   ���SAE��L1����200����Ԫ����ôcost�ļ������softmax��200�����˼һ�Ҫ���򴫲��������ˣ������ܶ������
%   �������µ�UFLDL��stl�̳���ֱ������unlabeled����ѧϰ������û�г�ȥ5-9������ѧ�����������ڷ���Ӧ�ø���Ч
%
%% ======================================================================
%  STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.

inputSize  = 28 * 28; % �����
numLabels  = 5;
hiddenSize = 200;       % ���ز�
sparsityParam = 0.1; % desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		             %  in the lecture notes). 
lambda = 3e-3;       % weight decay parameter       
beta = 3;            % weight of sparsity penalty term   
maxIter = 400;      % �������������ݶ��½����ٴ�

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
% ����SAEѧϰ������input��L1��
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
% ǰ�򴫲�����ר��д����������ʵ����: a = sigmoid(W1*data+repmat(b1,[1,size(data,2)]));
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
% ������ΪL1�㣬output��ɷ���Ľ����������0~4����labels���мල��
lambda = 1e-4;
inputSize = hiddenSize;
numClasses = numel(unique(trainLabels));%uniqueΪ�ҳ������еķ��ظ�Ԫ�ز���������

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
