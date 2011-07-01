function [] = Test_codes(name, n_train, n_test)
disp(name);
load(n_train);
Train.sample = fea;
Train.label = gnd;
load(n_test);
Test.sample = fea;
Test.label = gnd;

% disp('1--KNN');
% [predict, accuracy] = Knn(Train, Test, 3);
% disp(['The KNN accuracy: ', num2str(accuracy)]);
% disp('---------');
% 
% disp('2--Naive_Bayes (with Gaussian)');
% [predict, accuracy] = Naive_Bayes(Train, Test);
% disp(['The Naive_Bayes (no smoothing) accuracy: ', num2str(accuracy)]);
% disp('---------');
% 
% disp('3--Naive_Bayes (with smoothing)');
% [predict, accuracy] = Naive_Bayes_smooth(Train, Test, 1);
% disp(['The Naive_Bayes (with smoothing) accuracy: ', num2str(accuracy)]);
% disp('---------');

disp('4--Ridge_Regression');
[predict, accuracy] = Ridge_Reg(Train, Test, 0.5);
disp(['The Ridge_Regression accuracy: ', num2str(accuracy)]);
disp('---------');

end