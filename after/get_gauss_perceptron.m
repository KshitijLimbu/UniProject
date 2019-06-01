trainingSamples = 1000;
testingSamples = 100000;
[trainingData, trainingTarget] = GenerateGaussianDataLS(trainingSamples);
[testingData, testingTarget] = GenerateGaussianDataLS(testingSamples);
