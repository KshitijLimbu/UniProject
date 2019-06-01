trainingSamples = 1000;
testingSamples = 100000;
[trainingData, trainingTarget] = GenerateGaussianData(trainingSamples);
[testingData, testingTarget] = GenerateGaussianData(testingSamples);
