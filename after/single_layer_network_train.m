function [optimal_weight] = single_layer_network_train(input, targets, no_of_output, activation_function, error_function, error_val, learning_rate)


%activation_functions_permissible = {'logistic', 'softmax'};

%weights initialised by random selection from a zero mean, unit variance
%isotropic Gaussian where the variance is scaled by the fan_in of the
%hidden or output units
first_layer_weight = (randn(size(input, 2), no_of_output)/sqrt(size(input, 2) + 1))';
bias_weight =  randn(1, no_of_output)/sqrt(size(input, 2) + 1);

ndata = size(input, 1);

weights = [bias_weight' first_layer_weight]';
% randn('state', s) : the seed for the random weight initialisation can be
% set by calling this

err = 0;
total_err = 0;

bias_appended_input = [ones(ndata, 1), input];

count_incorrect_pred = 0;
total_pred_error = 10;
count_error = 0;
while total_pred_error > 0
    
    %input is in columns not rows!!!!
    %forward_propagation = input* first_layer_weight + ones(ndata, 1)* bias_weight;
    
    for n= 1:ndata
        
        switch activation_function
            case 'perceptron'
                forward_propagation = bias_appended_input(n,:)*weights;

                %hard_limiter
                %signum
                
                if forward_propagation > 0
                    out = 1;
                else
                    out = -1;
                end
                
                 if out == 1
                    output = 1;
                else
                    output = 0;
                 end
                
                target = targets(n,1);
                
            case 'adaline'
                
                 forward_propagation = bias_appended_input(n,:)*weights;
                      
                %linear
                
                if forward_propagation > 0
                    output = 1;
                else
                    output = 0;
                end
                                     
                target = targets(n,1);
                
            case 'single output'
                %sigmoid
                forward_propagation = bias_appended_input(n,:)*weights;
                
                y = 1./(1 + exp(-forward_propagation));
                
                target = targets(n);
                
                if y < 0.5
                    output = 0;
                else
                    output = 1;
                end
                
            case 'multi output'
                %softmax
                
                forward_propagation = bias_appended_input(n,:)*weights;
                
                temp = exp(forward_propagation);
                
                y = temp./(sum(temp,2)*ones(1,no_of_output));
                
                [t_output ind_t_output] = max(y);
                
                output = ind_t_output;
                
                [t indTarget] = max(targets(n,:));
                
                target = indTarget;
                
        end
        
        if target ~= output
            
            switch error_function
                
                case 'perceptron'
                    
                    delta_target = (targets(n,1) - output);
                    in = bias_appended_input(n,:)';
                    delta_weight_val = learning_rate*in*delta_target;
                             
                    weights = weights + delta_weight_val;
                    count_incorrect_pred = count_incorrect_pred + 1;
                 
                case 'adaline'
                    
                    delta_target = (targets(n,1) - forward_propagation);
                    in = bias_appended_input(n,:)';
                    delta_weight_val = learning_rate*in*delta_target;
                             
                    weights = weights + delta_weight_val;
                    count_incorrect_pred = count_incorrect_pred + 1;
                    
                case 'sum of square'
                    
                case 'cross entropy s'
                    
                    %calculating error error
                    %cross entropy error function
                    err = -sum(sum(targets(n).*log(y) + (1-targets(n)).*log(1-y)));
                    
                    %delta_weight_val = (learning_rate*edata*bias_appended_input(n,:))';
                    %add the derivative of the sigmoid function --> a(1-a)
                    
                    %error update
                    
                    delta_weight_val = (learning_rate*(targets(n) - y).*bias_appended_input(n,:))';
          
                    weights = weights + delta_weight_val;
                    
                    count_incorrect_pred = count_incorrect_pred + 1;
                    
                case 'cross entropy m'
                    
                    %error
                    %err = -sum(targets(n,:).*log(y));
                    
                    %total_err = err + total_err;
                    %weight update
                   
                    delta_target = (targets(n,:) - y);
                    in = bias_appended_input(n,:)';
                    delta_weight_val = learning_rate*in*delta_target;
                    
                    weights = weights + delta_weight_val;
                    
                    %tally
                    count_incorrect_pred = count_incorrect_pred + 1;
                    
            end
            
        end
        
    end
    
    total_pred_error = count_incorrect_pred/ndata;
    
    errorr = total_err/count_incorrect_pred;
    
    fprintf('Prediction Error value: %d \n', total_pred_error);
    fprintf('solo error: %d \n', count_error);
    fprintf('combined error: %d \n', errorr);
    count_incorrect_pred = 0;
    count_error = 0;
end

optimal_weight = weights;


