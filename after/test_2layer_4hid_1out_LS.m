trainingSamples = 1000;
testingSamples = 100000;
[trD, trT] = GenerateGaussianDataLS(trainingSamples);
[testingData, testingTarget] = GenerateGaussianDataLS(testingSamples);

matlab_2layer_4hid_1out;

[first_layer, second_layer] = two_layer_network_train(trD',trT' ,1, 4, 'sigmoid', 'cross entropy s', 'back-prop', 'batch', 0.2, 0.005)

[ incorrect_prediction ] = two_layer_network_test(first_layer, second_layer, testingData', testingTarget', 1)
