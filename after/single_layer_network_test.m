function [ error ] = single_layer_network_test(weight, testingData,testingTarget, no_of_output, algorithm)

ndata = size(testingData, 1);
bias_appended_input = [ones(ndata, 1), testingData];
weights = weight;

incorrect_pred = 0;


for n= 1:ndata
    
    switch algorithm
        case 'perceptron'
            forward_propagation = bias_appended_input(n,:)*weights;
            
            a = sum(forward_propagation);
            
            if a < 0
                output = 0;
            else
                output = 1;
            end
            
            target = testingTarget(n,1);
            
        case 'logistic'
            
            forward_propagation = bias_appended_input(n,:)*weights;
                       
            y = 1./(1 + exp(-forward_propagation));
            
            target = testingTarget(n,1);
            
            if y < 0.5
                output = 0;
            else
                output = 1;
            end
            
        case 'softmax'
            
            forward_propagation = bias_appended_input(n,:)*weights;
           
            temp = exp(forward_propagation);
            
            y = temp./(sum(temp,2)*ones(1,no_of_output));
            
            [t_output ind_t_output] = max(y);
            
            output = ind_t_output;
            
            [t indTarget] = max(testingTarget(n,:));
            
            target = indTarget;
            
    end
    
    if target ~= output
        
        incorrect_pred = incorrect_pred + 1;
        
        
    end
    
    
end

%error = (incorrect_pred/ndata) * 100;
error = incorrect_pred;
end

