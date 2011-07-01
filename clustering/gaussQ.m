function [ prob ] = gaussQ( X, Mu, Sigma )
%GAUSSQ Summary of this function goes here
%   Detailed explanation goes here
% X :  N x D matrix with N datapoints of D dimensions
% Mu:  K x D matrix with K GMM components of D dimensions
% Sigma: D x D x K covariance matrices of K components
% Output
%   prob: N x K matrix with N prob of K components
[N, D] = size(X);
[K, ~] = size(Mu);
prob = zeros(N, K);
for i=1:K
    Data = X - repmat(Mu(i,:),N,1);
    %[Sigma(:,:,i),~,~] = mPCA(Sigma(:,:,i), 0.8);
    prob(:,i) = sum(Data*inv(Sigma(:,:,i)).*Data, 2);
    prob(:,i) = exp(-0.5*prob(:,i)) / sqrt((2*pi)^D*(abs(det(Sigma(:,:,i)))));
end

end

