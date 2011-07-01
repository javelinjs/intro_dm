function [ predict, accuracy ] = mgmm( X, K, Test_Labels, pcathres )
%MGMM Summary of this function goes here
%   Detailed explanation goes here
% X: N x D data matrix
% K: # of components

threshold = 1e-15;
tic;
[X,~,~] = mPCA(X, pcathres);
[N, D] = size(X);
% randomly pick centroids
rndp = randperm(N);
centroids = X(rndp(1:K), :);

%init params
pMiu = centroids; % K x D
pPi = zeros(1, K); % 1 x K
pSigma = zeros(D, D, K); % D x D x K
% find the minimun dist and assign X to each centroids
pMiu_temp = pMiu';
distabs = abs(repmat(X,1,K)-repmat(pMiu_temp(:)',N,1));
distmat = zeros(N, K);
for k=1:K
    distmat(:,k)=sum(distabs(:,((k-1)*D+1):(k*D)),2);
end
[~, labels] = min(distmat, [], 2);
for k=1:K
    Xk = X(labels==k, :);  
    pPi(k) = size(Xk, 1)/N;
    pSigma(:,:,k) = cov(Xk);
end
% now use EM to get pPi, pMu, pSigma
while true
    pPi_prev = pPi;
    Px = gaussQ(X, pMiu, pSigma); % N x K matrix with N prob of K components
    Qz = Px .* repmat(pPi,N,1); % N x 1 vector with N 
    Qz = Qz ./ repmat(sum(Qz,2), 1, K);

    for k=1:K
        sumQ = sum(Qz(:,k),1);
        pPi(1,k) = sumQ/N;
        pMiu(k,:) = sum(X.*repmat(Qz(:,k),1,D),1)/sumQ;
        X_miu = X - repmat(pMiu(k,:),N,1);
        pSigma(:,:,k) = Qz(1,k)*(X_miu(1,:)'*X_miu(1,:));
        for j=2:N
            pSigma(:,:,k) = pSigma(:,:,k) + Qz(j,k)*(X_miu(j,:)'*X_miu(j,:));
        end
        pSigma(:,:,k) = pSigma(:,:,k)/sumQ; 
    end
    
    L = sum(abs(pPi-pPi_prev));
    if L < threshold
        break;
    end
end

[~,predict] = max(Qz, [], 2);
toc;
%calculate the accuracy
% [~,cluster_ind] = unique(Test_Labels);
% for k=1:length(cluster_ind)
%     ind = find(predict-predict(cluster_ind(k)) == 0);
%     predict(ind) = Test_Labels(cluster_ind(k));
% end
predict = bestMap(Test_Labels, predict);
accuracy = length(find(predict - Test_Labels ==0))/length(Test_Labels);

end

