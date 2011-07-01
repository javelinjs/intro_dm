function [ predict, accuracy ] = Naive_Bayes_smooth( Train, Test, alpha )
%NAIVE_BAYES_SMOOTH Summary of this function goes here
%   Detailed explanation goes here
Train_sample = Train.sample;
Train_label = Train.label;
Test_sample = Test.sample;
Test_label = Test.label;
Class_num = length(unique(Train_label)); %how many classes
Feature_num = size(Train_sample,2); %how many features per data
Para_likelihood = cell(2,Class_num); %1 for the larger split, 2 for the smaller
Sample_byclass = cell(1,Class_num);%Reorder the data set by class
Prior_prob = zeros(1,Class_num);%Prior probability of each class
starttime = cputime;
for i=1:1:size(Train_sample,1) %length(Train_sample) 
    % organize the data for each class, one row per data
    Sample_byclass{1,Train_label(i,1)} = [Sample_byclass{1,Train_label(i,1)}; Train_sample(i,:)];
    Prior_prob(1,Train_label(i,1)) = Prior_prob(1,Train_label(i,1)) + 1;
end
Prior_prob = Prior_prob/size(Train_sample,1); % Prior probability
%calculate the mean as the split
Feature_split=mean(Train_sample);
for i=1:1:Class_num  %model parameter
    Para_likelihood{1,i} = zeros(1,Feature_num);
    Para_likelihood{2,i} = zeros(1,Feature_num);
    %calculate the likelihood for each feature
    for j=1:size(Sample_byclass{1,i},2)
        sorted_f = sort(Sample_byclass{1,i}(:,j));
        %use (# of larger or smaller / sum) as the probability 
        larger = find(sorted_f > Feature_split(1,j));
        smaller = find(sorted_f <= Feature_split(1,j));
        %use the log-likelihood
        Para_likelihood{1,i}(1,j) = log(length(larger)+alpha)-log(length(sorted_f)+alpha);
        Para_likelihood{2,i}(1,j) = log(length(smaller)+alpha)-log(length(sorted_f)+alpha);
    end
end
endtime = cputime;
disp(['Train time: ', num2str(endtime-starttime), ' seconds']);
startime = cputime;
predict = [];
for i=1:size(Test_sample, 1)   %length(Test_sample)
     prob = log(Prior_prob);
     for j=1:Class_num
         hei=0;
         % for each feature, calculate the likelihood
         for k=1:1:Feature_num 
             if (Test_sample(i,k) > Feature_split(1,k)) %the bigger split
                 hei = hei + Para_likelihood{1,j}(1,k);
             else
                 hei = hei + Para_likelihood{2,j}(1,k);
             end
         end  %end for each feature
         prob(1,j) = prob(1,j)+hei;
     end %end for each class
     [value index] = max(prob);
     predict = [predict ; index];
end
endtime = cputime;
disp(['Test time: ', num2str(endtime-starttime), ' seconds']);
accuracy = length(find(predict - Test_label ==0))/length(Test_label);

%end

