function [ error ] = two_layer_network_test(weight1, weight2, testingData, testingTarget,no_of_output)

ndata = size(testingData, 1);
bias_appended_input = [ones(ndata, 1), testingData];
weights_first_layer = weight1;
weights_second_layer = weight2;
incorrect_pred = 0;

if no_of_output == 1
    out = 'single';
else
    out = 'multi';
end

for n= 1:ndata
    
    %forward propagation
    forward_propagation_hidden = bias_appended_input(n,:)*weights_first_layer;
    % sigmoid function
    forward_propagation_activation = 1./(1 + exp(-forward_propagation_hidden));
    %add bias
    forward_propagation_bias = [ones(1,1), forward_propagation_activation];
    

    
    switch out
                  
        case 'single'
            
            forward = forward_propagation_bias*weights_second_layer;
            
            
            y = 1./(1 + exp(-forward));
            
            target = testingTarget(n,1);
            
            if y < 0.5
                output = 0;
            else
                output = 1;
            end
            
        case 'multi'
            
            %forward_propagation = bias_appended_input(n,:)*weights;
            forward = forward_propagation_bias*weights_second_layer;
            
        
            temp = exp(forward);
            
            %y(y<realmin) = realmin;
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

error = incorrect_pred;

end

