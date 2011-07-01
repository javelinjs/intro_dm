function [predict, accuracy] = Naive_Bayes(Train, Test)
%Input: Training set and testing set, each row represents a instance, last column is label( begins from zero)
%Output:predict label by NaiveBayes as well as its accuracy
Train_sample = Train.sample;
Train_label = Train.label;
Test_sample = Test.sample;
Test_label = Test.label;
Class_num = length(unique(Train_label)); %how many classes
Feature_num = size(Train_sample,2); %how many features per data
Para_mean = cell(1,Class_num);%Mean for each feature and class
Para_dev = cell(1,Class_num);%Dev for each feature and class
Sample_byclass = cell(1,Class_num);%Reorder the data set by class
Prior_prob = zeros(1,Class_num);%Prior probability of each class
starttime = cputime;
for i=1:1:size(Train_sample,1) %length(Train_sample) 
    % organize the data for each class, one row per data
    Sample_byclass{1,Train_label(i,1)} = [Sample_byclass{1,Train_label(i,1)}; Train_sample(i,:)];
    Prior_prob(1,Train_label(i,1)) = Prior_prob(1,Train_label(i,1)) + 1;
end
Prior_prob = Prior_prob/size(Train_sample,1); % Prior probability
for i=1:1:Class_num  %model parameter
     miu = mean(Sample_byclass{1,i});
     delta = std(Sample_byclass{1,i});   
     Para_mean{1,i} = miu;
     Para_dev{1,i} = delta;
end
endtime = cputime;
disp(['Train time: ', num2str(endtime-starttime), ' seconds']);
starttime = cputime;
predict = [];
for i=1:size(Test_sample, 1)   %length(Test_sample)
     prob = log(Prior_prob);
     %hei=0;
     for j=1:Class_num
         hei = 0;
         for k=1:1:Feature_num
             %如果方差为0，调整
             if Para_dev{1,j}(1,k) == 0
                 Para_dev{1,j}(1,k) = 0.1667;
             end
             %calculate the log Gaussian distribution, 
             %suppose the features are i.i.d
             hei=hei-(Test_sample(i,k)-Para_mean{1,j}(1,k))^2/(2*Para_dev{1,j}(1,k)^2) ...,
                  -log(sqrt(2*pi)*Para_dev{1,j}(1,k));
             
         end  %end for each feature
         prob(1,j) = prob(1,j)+hei;
     end %end for each class
     [value index] = max(prob);
     predict = [predict ; index];
end
endtime = cputime;
disp(['Test time: ', num2str(endtime-starttime), ' seconds']);
accuracy = length(find(predict - Test_label ==0))/length(Test_label);