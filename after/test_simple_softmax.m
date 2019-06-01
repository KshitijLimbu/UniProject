trainingSamples = 1000;
testingSamples = 100000;
[trainingData, trainingTarget] = GenerateGaussianDataMulti(trainingSamples);
[testingData, testingTarget] = GenerateGaussianDataMulti(testingSamples);

[optimised_weights] = single_layer_network_train(trainingData', trainingTarget', 3, 'multi output', 'cross entropy m', 0.005, 0.2)

[ incorrect_predictions ] = single_layer_network_test(optimised_weights, testingData',testingTarget', 3, 'softmax')