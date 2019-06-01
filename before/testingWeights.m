function [ error ] = testingWeights(testingData, testingTarget, weightOne, weightTwo, threshold  )

a = size(testingData);

b = a(1,2); %no of iteration

x=1;
z = 0; % threshold
fail=0; 

     
while ( x < b+1)

     % 1 is the bias value,
     
 sumInput= testingData(1,x)*weightOne + testingData(2,x)*weightTwo - threshold ;
 
    
     
     if (sumInput > 0)
         
         z = 1;
                
     end
     
    
     
     if (testingTarget(1,x) ~= z)
         
         fail=fail+1;
               
     end
    
z=0;
    x=x+1; %iterates up to the number of training data
    
    
end

error = (fail/b) * 100;

end

