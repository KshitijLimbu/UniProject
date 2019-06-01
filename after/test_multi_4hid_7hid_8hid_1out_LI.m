trainingSamples = 1000;
testingSamples = 10000;
[trD, trT] = GenerateGaussianDataLI(trainingSamples);
[testingData, testingTarget] = GenerateGaussianDataLI(testingSamples);

matlab_multi_4_7_8_1out;

trT(2,:) = [];
net = [4 7 8];
[weight] = multi_layer_network_train(trD',trT' , 1, net, 'sigmoid', 'cross entropy s', 'back-prop','online', 0.5, 0.005);

[incorrect_prediction]  = multi_layer_network_test(weight, testingData', testingTarget', 1,2)

