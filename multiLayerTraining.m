function [weightOne,weightTwo] = multiLayerTraining( trainingData, trainingTarget, weightOneIn, weightTwoIn, learningRate )

a = size(trainingData);

weightOne = weightOneIn;
weightTwo = weightTwoIn;

epoch = 0;
while epoch<10000
    
inputMatrix = transpose(trainingData);
hiddenLayer=inputMatrix*weightOne;

%sigmoid function
activatedFunHidden= 1./(1+exp(-hiddenLayer));

outputLayer=activatedFunHidden*weightTwo;

activatedFunOutput= 1./(1+exp(-outputLayer)); %eqn

target=transpose(trainingTarget);
target=target(:,1);

error = target - activatedFunOutput; %eqn
actFunUpdateTwo= activatedFunOutput.*(1-activatedFunOutput); %eqn

actFunUpdateOne= activatedFunHidden.*(1-activatedFunHidden); %eqn1

% errorSqr = error.^2;
% errorSqr = sum(errorSqr);
% errorSqr = (1/2).*errorSqr;

%d(3) reused in weightOne update
updateVal= error .* actFunUpdateTwo;
updateValTrans=transpose(updateVal);

updateWeightTwoVal= updateValTrans * activatedFunHidden;
updateWeightTwoVal = transpose(updateWeightTwoVal);

weightTwo=weightTwo + learningRate*(updateWeightTwoVal);

inputMatrixTrans = transpose(inputMatrix);

z=transpose(weightTwo);
a=updateVal*z;
%b=transpose(a);
c=(a.*(actFunUpdateOne));
updateWeightOneVal = inputMatrixTrans*c;

weightOne = weightOne + learningRate*(updateWeightOneVal);

epoch=epoch+1;

% disp(weightOne);
% disp(weightTwo);
end








