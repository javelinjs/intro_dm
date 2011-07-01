function [predict, accuracy] = mkmeans(x, k, Test_Labels)

% INPUTS: X is the n x d matrix of data,
% where each row indicates an observation. K indicates
% the number of desired clusters. centroids is a k x d matrix for the
% initial cluster centers. 
% centers will be randomly chosen from the observations.
%
% OUTPUTS: predict provides a set of n indexes indicating cluster
% membership for each point. 
tic;
[n,d] = size(x);

% Pick some observations to be the cluster centers.
ind = ceil(n*rand(1,k));
centroids = x(ind,:);

% set up storage
% integer 1,...,k indicating cluster membership
predict = zeros(1,n); 
% Make this different to get the loop started.
oldpredict = ones(1,n);
% Set up maximum number of iterations.
maxiter = 1000;
iter = 1;
while ~isequal(predict,oldpredict) && iter < maxiter
    % Implement the hmeans algorithm
    oldpredict = predict;
    % For each point, find the distacentroidse to all cluster centers
	for i = 1:n
        dist = sum((repmat(x(i,:),k,1)-centroids).^2,2);
        [~,ind] = min(dist); % assign it to this cluster center
        predict(i) = ind;
    end
	% Find the new cluster centers
	for i = 1:k
        % find all points in this cluster
        ind = find(predict==i);
        % find the centroid
        centroids(i,:) = mean(x(ind,:));
	end
	iter = iter + 1;
end
predict = predict';
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