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
% 这里，特征数量就是隐藏层神经元数量，也就是某篇博客说的Feature Map的kernel的数量（有多少个特征kernel，用来卷积）
assert(numFeatures == size(W,1), 'W should have numFeatures rows');
numImages = size(images, 4);%第4维的大小，即图片的样本数
imageDim = size(images, 1);%第1维的大小,即图片的行数
imageChannels = size(images, 3);%第3维的大小，即图片的通道数
% size(W, 2)就是input的数量 y = AX, input的数量是patchSize * imageChannels
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

% 这里目的就是对input进行整理
WT = W*ZCAWhite;%等效的网络参数
b_mean = b - WT*meanPatch;%针对未均值化的输入数据需要加入该项

% --------------------------------------------------------
% 第几个feature，每个hidden神经元对应一个Feature，这里之前训练好了，也就是特征提取，这里的卷积后图像是 
% （imageDim - patchDim + 1） * （imageDim - patchDim + 1）的，把3通道直接加起来了
convolvedFeatures = zeros(numFeatures, numImages, imageDim - patchDim + 1, imageDim - patchDim + 1);
for imageNum = 1:numImages
  for featureNum = 1:numFeatures

    % convolution of image with feature matrix for each channel
    convolvedImage = zeros(imageDim - patchDim + 1, imageDim - patchDim + 1);
    for channel = 1:imageChannels

      % Obtain the feature (patchDim x patchDim) needed during the convolution
      % ---- YOUR CODE HERE ----
      offset = (channel-1)*patchSize;
      feature = reshape(WT(featureNum,offset+1:offset+patchSize), patchDim, patchDim);%取一个权值图像块出来
%       im  = images(:,:,channel,imageNum);

      % Flip the feature matrix because of the definition of convolution, as explained later
      % 因为卷积，需要旋转一下，然后把它们放在前两维      
      feature = flipud(fliplr(squeeze(feature)));
      
      % Obtain the image
      im = squeeze(images(:, :, channel, imageNum));%取一张图片出来，并挪到前两维，要不无法处理

      % Convolve "feature" with "im", adding the result to convolvedImage
      % be sure to do a 'valid' convolution
      % ---- YOUR CODE HERE ----
      convolvedoneChannel = conv2(im, feature, 'valid');% 这部很简单，就是卷积，valid是只对有效图像进行卷积，会缩小图像
%      卷积后图像。 这里还不大明白，这样真的科学吗？
      convolvedImage = convolvedImage + convolvedoneChannel;%直接把3通道的值加起来，类似直接对彩色图像卷积吧。
      
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