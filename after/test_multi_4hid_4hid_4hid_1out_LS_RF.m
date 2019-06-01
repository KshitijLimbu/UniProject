trainingSamples = 1000;
testingSamples = 10000;
[trD, trT] = GenerateGaussianDataLS(trainingSamples);
[testingData, testingTarget] = GenerateGaussianDataLS(testingSamples);

matlab_multi_4_4_4_1out;

trT(2,:) = [];
net = [4 4 4];
[weight] = multi_layer_network_train(trD',trT' , 1, net, 'sigmoid', 'cross entropy s', 'feedback-ali','online', 0.5, 0.005);

[incorrect_prediction]  = multi_layer_network_test(weight, testingData', testingTarget', 1,3)

