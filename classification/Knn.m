function [ predict, accuracy ] = Knn( Train, Test, K )
%KNN Summary of this function goes here
%   Detailed explanation goes here
Train_sample = Train.sample;
Train_label = Train.label; 
Test_sample = Test.sample;
Test_label = Test.label;
disp(['Train time: 0 seconds']);
%K = 3;
starttime = cputime;
% Now do the knn-predict
predict = [];
mdist = zeros(1, size(Train_sample, 1)); %container for distance
k_nearest = zeros(1, K);
for i=1:size(Test_sample, 1)   %length(Test_sample)
     for j=1:size(Train_sample, 1)
         mdist(j) = man_distance(Test_sample(i,:), Train_sample(j,:));
     end
     for j=1:K
        [value index] = min(mdist);
        k_nearest(j) = Train_label(index, 1); %the labels
     end
     % find which label appears the most 
     uni_kn = unique(k_nearest);
     n_kn = histc(k_nearest, uni_kn);
     [value label_index] = max(n_kn);
     predict = [predict ; uni_kn(label_index)];
end
endtime = cputime;
disp(['Test time: ', num2str(endtime-starttime), ' seconds']);
accuracy = length(find(predict - Test_label ==0))/length(Test_label);

end

