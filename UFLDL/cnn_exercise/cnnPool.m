function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(featureNum, imageNum, imageRow, imageCol)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(featureNum, imageNum, poolRow, poolCol)
%     

numImages = size(convolvedFeatures, 2);%图片数
numFeatures = size(convolvedFeatures, 1);%特征数
convolvedDim = size(convolvedFeatures, 3);%图片的行数
% 这里正好能除尽 57/19=3，不能除尽的时候怎么办，应该是提前计算好的pool kernel的大小
resultDim  = floor(convolvedDim / poolDim);
pooledFeatures = zeros(numFeatures, numImages, resultDim, resultDim);

% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   numFeatures x numImages x (convolvedDim/poolDim) x (convolvedDim/poolDim) 
%   matrix pooledFeatures, such that
%   pooledFeatures(featureNum, imageNum, poolRow, poolCol) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region 
%   (see http://ufldl/wiki/index.php/Pooling )
%   
%   Use mean pooling here.
% -------------------- YOUR CODE HERE --------------------
for imageNum = 1:numImages
    for featureNum = 1:numFeatures
        for poolRow = 1:resultDim
            offsetRow = 1+(poolRow-1)*poolDim;
            for poolCol = 1:resultDim
                offsetCol = 1+(poolCol-1)*poolDim;
                patch = convolvedFeatures(featureNum,imageNum,offsetRow:offsetRow+poolDim-1,...
                    offsetCol:offsetCol+poolDim-1);%取出一个patch
                pooledFeatures(featureNum,imageNum,poolRow,poolCol) = mean(patch(:));%使用均值pool
            end
        end
    end
end

end