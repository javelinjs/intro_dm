function [ predict, accuracy ] = Ridge_Reg( Train, Test, lambda )
%RIDGE_REG Summary of this function goes here
%   Detailed explanation goes here
Train_sample = Train.sample;
Train_label = Train.label;
Test_sample = Test.sample;
Test_label = Test.label;

label=1;

Class_num = length(unique(Train_label)); %how many classes
Y = zeros(size(Train_sample,1), Class_num);
%a_byclass = cell(1, Class_num);
%result = zeros(1,Class_num);
%Sample_byclass = cell(1,Class_num);%Reorder the data set by class
starttime = cputime;
for i=1:1:size(Train_sample,1) %length(Train_sample) 
    % organize the data for each class, one row per data
    %Sample_byclass{1,Train_label(i,1)} = [Sample_byclass{1,Train_label(i,1)}; Train_sample(i,:)];
    Y(i,Train_label(i,1)) = label; %if the ith data belongs to jth class, then Y(i,j)=label
end

% X = Sample_byclass{1,i};
% y = label+zeros(size(X,1),1); 
% [U,Sigma,V] = svd(Train_sample); %full svd
% Gamma = Sigma'*Sigma;
% Gamma = Gamma + lambda * eye(size(Gamma));
% %calculate the inverse of Gamma, which is diagonal
% for j=1:size(Gamma,1)
%     Gamma(j,j) = 1/Gamma(j,j);
% end
% A = V*Gamma*Sigma'*U'*Y;
%a_byclass{1,i} = V*Gamma*Sigma'*U'*y;
X = Train_sample;
if (size(X,1)>size(X,2))
    G = X'*X;
    I = lambda*eye(size(G));
    A=inv(G+I)*X'*Y;
else
    lambda = 1/lambda;
    G = X*X';
    I_d = eye(size(X,2));
    I_n = eye(size(X,1));
    inv_Sigma = lambda*I_d-lambda*X'*inv(I_n+(lambda*X)*X')*(lambda*X);
    A = inv_Sigma*X'*Y;
end
endtime = cputime;
disp(['Train time: ', num2str(endtime-starttime), ' seconds']);
starttime = cputime;

predict = [];
for i=1:size(Test_sample, 1)   %length(Test_sample)
    %reg = zeros(1,Class_num);
    reg = Test_sample(i,:)*A;
%     for j=1:Class_num
%         reg(1,j) = abs(Test_sample(i,:)*a_byclass{1,j} - label);
%     end %end for each class
    [value index] = max(reg);
    predict = [predict ; index];
end
endtime = cputime;
disp(['Test time: ', num2str(endtime-starttime), ' seconds']);
accuracy = length(find(predict - Test_label ==0))/length(Test_label);
end

