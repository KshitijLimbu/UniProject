function [weightOne, weightTwo] = trainingWeights( trainingData, desiredResult, weightOne,weightTwo,learningRate, threshold )

a = size(trainingData);

b = a(1,2); %no of iteration

x=1;
z = 0; % threshold
 
epoch = 0;
while epoch<100000

while ( x < b+1)

     % 1 is the bias value,
     
 sumInput= trainingData(1,x)*weightOne + trainingData(2,x)*weightTwo  - threshold ;
 
    
     
     if (sumInput > 0)
         
         z = 1;
                
     end
     
    
     
     if (desiredResult(1,x) ~= z)
         
         error= desiredResult(1,x)-z;
        
         updatedWOne = trainingData(1,x) * error * learningRate;
         
         updatedWTwo = trainingData(2,x) * error * learningRate;
         
%          updateBias=  bias * error * learningRate;
%          
%          bias = bias + updateBias;
         
         weightOne = weightOne + updatedWOne;
         weightTwo = weightTwo + updatedWTwo;
              
     end
    
z=0;
    x=x+1; %iterates up to the number of training data
    
    
end

x=1;
 
    
epoch = epoch +1;
end


