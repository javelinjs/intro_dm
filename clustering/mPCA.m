function [ signals,PC,lambda ] = mPCA( X, threshold )
%MPCA Summary of this function goes here
%   Detailed explanation goes here
% N x D data, where D refers to the dimension, N refers to the # of data
%threshold = 0.1;
% calculate the covariance matrix 
% [N,D] = size(X);
% mn = mean(X); 
% X = X - repmat(mn,N,1); 
% S = 1 / (N-1) * X' * X;
S = cov(X);
% find the eigenvectors and eigenvalues 
[PC, lambda] = eig(S); 
% extract diagonal of matrix as vector 
lambda = diag(lambda);
% sort the variances in decreasing order
[~, rindices] = sort(-1*lambda);
lambda = lambda(rindices); PC = PC(:,rindices);
% the positive lambdas
rindices = find(lambda>0);
lambda = lambda(rindices); PC = PC(:,rindices);
% look for the first k lambda that sum(lambda(1:k))/lambda_sum>threshold
lambda_sum = sum(lambda);
lambda_k = 0;
for i=1:length(lambda)
    lambda_k = lambda_k + lambda(i);
    if lambda_k/lambda_sum > threshold
        break;
    end
end
lambda = lambda(1:i); PC = PC(:,1:i);
% project the original data set 
signals = X * PC;

end