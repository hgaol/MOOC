function convolvedFeatures = cnnConvolve(patchDim, numFeatures, images, W, b, ZCAWhite, meanPatch)
%cnnConvolve Returns the convolution of the features given by W and b with
%the given images
%
% Parameters:
%  patchDim - patch (feature) dimension
%  numFeatures - number of features
%  images - large images to convolve with, matrix in the form
%           images(r, c, channel, image number)
%  W, b - W, b for features from the sparse autoencoder
%  ZCAWhite, meanPatch - ZCAWhitening and meanPatch matrices used for
%                        preprocessing
%
% Returns:
%  convolvedFeatures - matrix of convolved features in the form
%                      convolvedFeatures(featureNum, imageNum, imageRow, imageCol)

patchSize = patchDim*patchDim;
% ������������������ز���Ԫ������Ҳ����ĳƪ����˵��Feature Map��kernel���������ж��ٸ�����kernel�����������
assert(numFeatures == size(W,1), 'W should have numFeatures rows');
numImages = size(images, 4);%��4ά�Ĵ�С����ͼƬ��������
imageDim = size(images, 1);%��1ά�Ĵ�С,��ͼƬ������
imageChannels = size(images, 3);%��3ά�Ĵ�С����ͼƬ��ͨ����
% size(W, 2)����input������ y = AX, input��������patchSize * imageChannels
assert(patchSize*imageChannels == size(W,2), 'W should have patchSize*imageChannels cols');

% Instructions:
%   Convolve every feature with every large image here to produce the 
%   numFeatures x numImages x (imageDim - patchDim + 1) x (imageDim - patchDim + 1) 
%   matrix convolvedFeatures, such that 
%   convolvedFeatures(featureNum, imageNum, imageRow, imageCol) is the
%   value of the convolved featureNum feature for the imageNum image over
%   the region (imageRow, imageCol) to (imageRow + patchDim - 1, imageCol + patchDim - 1)
%
% Expected running times: 
%   Convolving with 100 images should take less than 3 minutes 
%   Convolving with 5000 images should take around an hour
%   (So to save time when testing, you should convolve with less images, as
%   described earlier)

% -------------------- YOUR CODE HERE --------------------
% Precompute the matrices that will be used during the convolution. Recall
% that you need to take into account the whitening and mean subtraction
% steps

% ����Ŀ�ľ��Ƕ�input��������
WT = W*ZCAWhite;%��Ч���������
b_mean = b - WT*meanPatch;%���δ��ֵ��������������Ҫ�������

% --------------------------------------------------------
% �ڼ���feature��ÿ��hidden��Ԫ��Ӧһ��Feature������֮ǰѵ�����ˣ�Ҳ����������ȡ������ľ����ͼ���� 
% ��imageDim - patchDim + 1�� * ��imageDim - patchDim + 1���ģ���3ͨ��ֱ�Ӽ�������
convolvedFeatures = zeros(numFeatures, numImages, imageDim - patchDim + 1, imageDim - patchDim + 1);
for imageNum = 1:numImages
  for featureNum = 1:numFeatures

    % convolution of image with feature matrix for each channel
    convolvedImage = zeros(imageDim - patchDim + 1, imageDim - patchDim + 1);
    for channel = 1:imageChannels

      % Obtain the feature (patchDim x patchDim) needed during the convolution
      % ---- YOUR CODE HERE ----
      offset = (channel-1)*patchSize;
      feature = reshape(WT(featureNum,offset+1:offset+patchSize), patchDim, patchDim);%ȡһ��Ȩֵͼ������
%       im  = images(:,:,channel,imageNum);

      % Flip the feature matrix because of the definition of convolution, as explained later
      % ��Ϊ�������Ҫ��תһ�£�Ȼ������Ƿ���ǰ��ά      
      feature = flipud(fliplr(squeeze(feature)));
      
      % Obtain the image
      im = squeeze(images(:, :, channel, imageNum));%ȡһ��ͼƬ��������Ų��ǰ��ά��Ҫ���޷�����

      % Convolve "feature" with "im", adding the result to convolvedImage
      % be sure to do a 'valid' convolution
      % ---- YOUR CODE HERE ----
      convolvedoneChannel = conv2(im, feature, 'valid');% �ⲿ�ܼ򵥣����Ǿ����valid��ֻ����Чͼ����о��������Сͼ��
%      �����ͼ�� ���ﻹ�������ף�������Ŀ�ѧ��
      convolvedImage = convolvedImage + convolvedoneChannel;%ֱ�Ӱ�3ͨ����ֵ������������ֱ�ӶԲ�ɫͼ�����ɡ�
      
      % ------------------------

    end
    
    % Subtract the bias unit (correcting for the mean subtraction as well)
    % Then, apply the sigmoid function to get the hidden activation
    % ---- YOUR CODE HERE ----

    convolvedImage = sigmoid(convolvedImage+b_mean(featureNum));
    
    
    % ------------------------
    
    % The convolved feature is the sum of the convolved values for all channels
    convolvedFeatures(featureNum, imageNum, :, :) = convolvedImage;
  end
end


end

function sigm = sigmoid(x)
    sigm = 1./(1+exp(-x));
end