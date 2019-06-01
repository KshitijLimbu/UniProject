% read testing dataset
% testing data filenames
imgFile = 't10k-images-idx3-ubyte';
labelFile = 't10k-labels-idx1-ubyte';
% start from beginning
offset=0;
% only want 1000 patterns
readDigits=10000;
% Read digits and labels from MNIST database by Sid H
[imgs TestLabels] = readMNIST(imgFile, labelFile, readDigits, offset);
TestData=[];
for idx =1:length(imgs)
 tmp = imgs(:,:,idx);
 TestData(idx,:) = tmp(:);
end

disp('Loaded testing dataset');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% read training dataset
% training data filenames
imgFile = 'train-images-idx3-ubyte';
labelFile = 'train-labels-idx1-ubyte';
% start from beginning
offset=0;
% only want 1000 patterns
readDigits=10000;
% Read digits and labels from MNIST database by Sid H
[imgs TrainLabels] = readMNIST(imgFile, labelFile, readDigits, offset);
TrainData=[];
for idx =1:length(imgs)
 tmp = imgs(:,:,idx);
 TrainData(idx,:) = tmp(:);
end
disp('Loaded training dataset');


% plot all iamges in training dataset
%PlotAllImages(TrainData,'Dataset', 10);