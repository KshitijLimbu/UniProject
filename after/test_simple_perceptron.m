trainingSamples = 1000;
testingSamples = 100000;
[trainingData, trainingTarget] = GenerateGaussianDataLS(trainingSamples);
[testingData, testingTarget] = GenerateGaussianDataLS(testingSamples);

[optimised_weights] = single_layer_network_train(trainingData', trainingTarget', 1, 'perceptron', 'perceptron', 0.005, 0.2)

[ incorrect_predictions ] = single_layer_network_test(optimised_weights, testingData',testingTarget', 1, 'perceptron')