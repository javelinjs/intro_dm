function [predict, accuracy] = spectral_clustering(X, k, Test_Labels)

[N, D] = size(X);
W=zeros(N, N);
for i=1:N
    for j=i+1:N
        dist = 1/sum(abs(X(i,:)-X(j,:)));
        W(i,j) = dist;
        W(j,i) = dist;
    end
end

tic;
D = diag(sum(W));
L = D-W;

opt = struct('issym', true, 'isreal', true);
[V dummy] = eigs(L, D, k, 'SM', opt);
[predict, accuracy] = mkmeans(V, k, Test_Labels);
toc;
end
