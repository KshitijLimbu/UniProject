trainingSamples = 1000;
testingSamples = 100000;
[trainingData, trainingTarget] = GenerateGaussianDataMulti(trainingSamples);
[testingData, testingTarget] = GenerateGaussianDataMulti(testingSamples);

matlab_2layer_4hid_3out;

[first_layer, second_layer] = two_layer_network_train(trainingData',trainingTarget' ,3, 4, 'softmax', 'sum of square', 'back-prop', 'online', 0.2, 0.05)

[ incorrect_prediction ] = two_layer_network_test(first_layer, second_layer, testingData', testingTarget', 3)
