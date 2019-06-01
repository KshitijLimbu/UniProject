trainingSamples = 1000;
testingSamples = 10000;
[trD, trT] = GenerateGaussianDataMulti(trainingSamples);
[testingData, testingTarget] = GenerateGaussianDataMulti(testingSamples);


matlab_multi_4_4_4_3out;

net = [4 4 4];
[weight] = multi_layer_network_train(trD',trT' , 3, net, 'softmax', 'cross entropy m', 'feedback-ali','online', 0.5, 0.005);

[incorrect_prediction]  = multi_layer_network_test(weight, testingData', testingTarget', 3,3)

