function [ errorRate ] = testMulti(testingData, testingTarget, weightOne, weightTwo  )

inputMatrix = transpose(testingData);
hiddenLayer=inputMatrix*weightOne;

%sigmoid function
activatedFunHidden= 1./(1+exp(-hiddenLayer));

outputLayer=activatedFunHidden*weightTwo;

activatedFunOutput= 1./(1+exp(-outputLayer)); %eqn

testingTarget=transpose(testingTarget);

for test = 1:length(testingTarget)
    if activatedFunOutput(test)<0.5
        activatedFunOutput(test) = 0;
    else
        activatedFunOutput(test) = 1;
    end
end

error = 0;
for test = 1:length(testingTarget)
    if testingTarget(test,1) ~= activatedFunOutput(test)
        error = error +1;
  
    end
end


errorRate = (error/test) * 100;



