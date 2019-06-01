function out = multi_layer_network_test(weights, testingData, testingTarget,no_of_output, no_of_layers) 

ndata = size(testingData, 1);
bias_appended_input = [ones(ndata, 1), testingData];

all_weights = weights;

count_incorrect_pred = 0;

if no_of_output == 1
    output_function = 'single';
else
    output_function = 'multi';
end

no_of_weights = no_of_layers + 1;

for n=1:ndata



        for a = 1:no_of_layers
            
            %forward propagation
            if a == 1
                
                forward_propagation_hidden = bias_appended_input(n,:)*all_weights.(strcat('weight',num2str(a)));
                % sigmoid function
                forward_propagation_activation = 1./(1 + exp(-forward_propagation_hidden));
                %add bias
                forward_propagation_bias = [ones(1,1), forward_propagation_activation];
                
                temp = strcat('f_prop',num2str(a));
                all_forward_propagation_bias.(temp) = forward_propagation_bias;
                all_forward_propagation.(temp) = forward_propagation_activation;
                
            else
                forward_propagation_hidden = all_forward_propagation_bias.((strcat('f_prop',num2str(a-1))))*all_weights.(strcat('weight',num2str(a)));
                forward_propagation_activation = 1./(1 + exp(-forward_propagation_hidden));
                forward_propagation_bias = [ones(1,1), forward_propagation_activation];
                
                temp = strcat('f_prop',num2str(a));
                all_forward_propagation_bias.(temp) = forward_propagation_bias;
                all_forward_propagation.(temp) = forward_propagation_activation;
                
            end
            %weight for first layer update
        end
         
        
        switch output_function
            
            case 'single'
                
                forward = all_forward_propagation_bias.((strcat('f_prop',num2str(no_of_layers))))*all_weights.(strcat('weight',num2str(no_of_weights)));
                y = 1./(1 + exp(-forward));
                
                target = testingTarget(n);
                
                if y < 0.5
                    output = 0;
                else
                    output = 1;
                end
                
            case 'multi'
                
                forward = all_forward_propagation_bias.((strcat('f_prop',num2str(no_of_layers))))*all_weights.(strcat('weight',num2str(no_of_weights)));
                
                temp = exp(forward);
                
                %y(y<realmin) = realmin;
                y = temp./(sum(temp,2)*ones(1,no_of_output));
                
                [t_output ind_t_output] = max(y);
                
                output = ind_t_output;
                
                [t indTarget] = max(testingTarget(n,:));
                
                target = indTarget;
                
        end
        
        
        if target ~= output
            
            count_incorrect_pred = count_incorrect_pred + 1;
            
           
            
        end
       
  
    
    
  
   
    
end


out = count_incorrect_pred;

end
