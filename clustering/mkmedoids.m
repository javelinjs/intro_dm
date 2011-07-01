function [predict, accuracy] = mkmedoids(x, k, Test_Labels)
tic;
[n,d] = size(x);
% Pick some observations to be the cluster centers.
ind = ceil(n*rand(1,k));
medroids = x(ind,:);

% set up storage
% integer 1,...,k indicating cluster membership
predict = zeros(1,n); 
% Make this different to get the loop started.
oldpredict = ones(1,n);
% The number in each cluster.
nr = zeros(1,k); 
% Set up maximum number of iterations.
maxiter = 1000;
iter = 1;
while ~isequal(predict,oldpredict) && iter < maxiter
    % Implement the hmeans algorithm
    oldpredict = predict;
    % For each point, find the distamedroidse to all cluster centers
	for i = 1:n
        dist = sum((repmat(x(i,:),k,1)-medroids).^2,2);
        [~,ind] = min(dist); % assign it to this cluster center
        predict(i) = ind;
    end
	% Find the new cluster centers
	for i = 1:k
        % find all points in this cluster
        ind = find(predict==i);
        cluster = x(ind, :);
        % Find the number in each cluster;
        nr(i) = length(ind);
        % find the medoid
        dist_sum = zeros(1, nr(i));
        for j=1:nr(i)
            dist = sum((repmat(cluster(j,:),nr(i),1)-cluster).^2,2);
            dist_sum(1,j) = sum(dist);
        end
        [~,index] = min(dist_sum);
        medroids(i,:) = cluster(index,:);      
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