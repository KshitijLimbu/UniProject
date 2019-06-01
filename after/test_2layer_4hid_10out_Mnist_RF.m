



%dummy variable
size_train = size(TrainLabels,1);
size_test = size(TestLabels,1);

ones_train = ones(size_train);
ones_test = ones(size_test);

ones_train = ones_train(:,1);
ones_test = ones_test(:,1);

change_train_label = TrainLabels + ones_train;
training_label = dummyvar(change_train_label);

change_test_label = TestLabels + ones_test;
testing_label = dummyvar(change_test_label);

matlab_multi_4_10out;

[first_layer, second_layer] = two_layer_network_train(TrainData,training_label ,10, 4, 'softmax', 'cross entropy m', 'feedback-ali', 'online', 0.2, 0.005)

[ incorrect_prediction ] = two_layer_network_test(first_layer, second_layer, TestData, testing_label, 10)

