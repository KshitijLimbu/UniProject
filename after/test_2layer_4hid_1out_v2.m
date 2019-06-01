trainingSamples = 1000;
testingSamples = 100000;
[trainingData, trainingTarget] = GenerateGaussianDataLI(trainingSamples);
[testingData, testingTarget] = GenerateGaussianDataLI(testingSamples);

matlab_2layer_4hid_1out;

[first_layer, second_layer] = two_layer_network_train(trainingData',trainingTarget' ,1, 4, 'sigmoid', 'cross entropy s', 'back-prop', 'online', 0.2, 0.05)

[ incorrect_prediction ] = two_layer_network_test(first_layer, second_layer, testingData', testingTarget', 1)
