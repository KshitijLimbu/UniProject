trainingSamples = 10000;
testingSamples = 1000;
[trD, trT] = GenerateGaussianDataLI2(trainingSamples);
[teD, teT] = GenerateGaussianDataLI2(testingSamples);

matlab_2layer_4hid_1out;

[ first_layer, second_layer ] = two_layer_network_train(trD',trT' ,1, 4, 'sigmoid', 'cross entropy s', 'feedback_ali', 'online', 0.2, 0.05)

[ incorrect_prediction ] = two_layer_network_test(first_layer, second_layer, teD', teT', 1)
