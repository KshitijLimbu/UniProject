function [first_layer, second_layer] = two_layer_network_train(input, targets, no_of_output, no_hidden_neuron, output_function,error_function, algorithm, learning, learning_rate, error_val)

ndata = size(input, 1);
%ntargets = size(targets, 2);


if no_of_output == 1
    out_funct = 'single';
else
    out_funct = 'multi';
end

first_layer_weight = (randn(size(input, 2), no_hidden_neuron)/sqrt(size(input, 2) + 1))';
first_layer_bias =  randn(1, no_hidden_neuron)/sqrt(size(input, 2) + 1);

second_layer_weight = (randn(no_hidden_neuron, no_of_output)/sqrt(no_hidden_neuron + 1))';
second_layer_bias = randn(1, no_of_output)/sqrt(no_hidden_neuron + 1);

weights_first_layer = [first_layer_bias' first_layer_weight]';


weights_second_layer = [second_layer_bias' second_layer_weight]';

% for batch gradient descent
[a b] = size(weights_first_layer);
total_gradient_first = zeros(a,b);
[a b] = size(weights_second_layer);
total_gradient_second = zeros(a,b);


% required for first layer update
uw = weights_second_layer;
uw(1,:) = [];

edata = 10;

bias_appended_input = [ones(ndata, 1), input];

count_incorrect_pred = 0;
total_pred_error = 10;
count_error = 0;

err = 10;

%total error
total_error = 0;

%multi_layer look into it

%random weights

if no_of_output == 1
random_weight = -.5+(.5+.5)*rand(1, no_hidden_neuron);

else
    [a,b] = size(second_layer_weight);
    random_weight = -.5+(.5+.5)*rand(b,a);
end

%batch training 1000 epochs
batch = 1;

epochs = 100;

while batch < epochs
    
    %input is in columns not rows!!!!
    %forward_propagation = input* first_layer_weight + ones(ndata, 1)* bias_weight;
    
    for n= 1:ndata
        
        %forward propagation
        forward_propagation_hidden = bias_appended_input(n,:)*weights_first_layer;
        % sigmoid function
        forward_propagation_activation = 1./(1 + exp(-forward_propagation_hidden));
        %add bias
        forward_propagation_bias = [ones(1,1), forward_propagation_activation];
        
        %weight for first layer update
        %this
        
        
        %unchanged_second_layer = uw;
        
        switch output_function
            
            case 'sigmoid'
                
                forward = forward_propagation_bias*weights_second_layer;
                
                y = 1./(1 + exp(-forward));
                
                target = targets(n,1);
                
                if y < 0.5
                    output = 0;
                else
                    output = 1;
                end
                
            case 'softmax'
                
                %forward_propagation = bias_appended_input(n,:)*weights;
                forward = forward_propagation_bias*weights_second_layer;
                
                temp = exp(forward);
                
                y = temp./(sum(temp,2)*ones(1,no_of_output));
                
                [t_output ind_t_output] = max(y);
                
                output = ind_t_output;
                
                [t indTarget] = max(targets(n,:));
                
                target = indTarget;
                
        end
        
        if target ~= output
            
            switch out_funct
                case 'single'
                    
                    switch algorithm
                        
                        case 'back-prop'
                            % back propagation
                            %second layer
                            delta_target = -(targets(n,1) - y);
                            sig_der_2 = y*(1-y);
                            delta_weight_second= delta_target*(sig_der_2)*forward_propagation_bias;
                            
                            %first layer
                            sig_der_1 = forward_propagation_activation.*(1-forward_propagation_activation);
                            
                            %delta_weight_first_1 = (delta_target*sig_der_2*(bias_appended_input(n,:))');
                            
                              %(bias_appended_input(n,:))'
                            %try
                            unchangedW = weights_second_layer;
                            unchangedW(1,:) = [];
                            
                            delta_weight_first_1 = (delta_target*sig_der_2* unchangedW );
                            
                          
                            
                            %delta_weight_first_2 = unchangedW.*sig_der_1';
                            
                            delta_weight_first_2 = delta_weight_first_1 .*sig_der_1';
                            
                            %bias_appended_input(n,:)'
                            delta_weight_first = bias_appended_input(n,:)'*delta_weight_first_2';
                            
                            switch learning
                                case 'batch'
                                    total_gradient_first = delta_weight_first + total_gradient_first;
                                    total_gradient_second = total_gradient_second + delta_weight_second';
                                case 'online'
                                    %second layer update
                                    weights_second_layer = weights_second_layer - (learning_rate*delta_weight_second');
                                    
                                    %first layer update
                                    weights_first_layer = weights_first_layer - (learning_rate*delta_weight_first);
                                    
                                    %uw = weights_second_layer;
                                    %uw(1,:) = [];
                                    
                            end
                            count_incorrect_pred = count_incorrect_pred + 1;
                            
                        case 'feedback-ali'
                            
                            %second layer
                            delta_target = -(targets(n) - y);
                            sig_der_2 = y*(1-y);
                            delta_weight_second= delta_target*(sig_der_2)*forward_propagation_bias;
                            
                            
                            weights_second_layer = weights_second_layer - (learning_rate*delta_weight_second');
                            
                            %first layer
                            
                            sig_der_1 = forward_propagation_activation.*(1-forward_propagation_activation);
                            
                            delta_weight_first_1 = (delta_target*(bias_appended_input(n,:))');
                            delta_weight_first_2 = (random_weight.*sig_der_1);
                            
                            delta_weight_first = delta_weight_first_1 * delta_weight_first_2;
                            
                            %delta_weight_first = delta_target * random_weight .* (bias_appended_input(n,:))' .* sig_der_1;
                            
                            weights_first_layer = weights_first_layer - (learning_rate * delta_weight_first);
                            
                            
                            
                            count_incorrect_pred = count_incorrect_pred + 1;
                            
                    end
                case 'multi'
                    
                    switch algorithm
                        
                        case 'back-prop'
                            % back propagation
                            %second layer
                            t = targets(n,:);
                            delta_target = y - t;
                            sig_der_2 = y.*(1-y);
                            delta_weight_second= forward_propagation_bias'* (delta_target.*(sig_der_2));
                      
                            %first layer
                            unchangedW = weights_second_layer;
                            unchangedW(1,:) = [];
                            
                            sig_der_1 = forward_propagation_activation.*(1-forward_propagation_activation);
                            
                            delta_weight_first_1 = unchangedW*(delta_target.*sig_der_2)';
                            
                            delta_weight_first = (bias_appended_input(n,:))'*(sig_der_1.*delta_weight_first_1');
                            
                            switch learning
                                case 'batch'
                                    total_gradient_first = delta_weight_first + total_gradient_first;
                                    total_gradient_second = total_gradient_second + delta_weight_second;
                                case 'online'
                                    %second layer update
                                    weights_second_layer = weights_second_layer - (learning_rate*delta_weight_second);
                                    
                                    %first layer update
                                    weights_first_layer = weights_first_layer - (learning_rate*delta_weight_first);
                                    
                                    %uw = weights_second_layer;
                                    %uw(1,:) = [];
                                    
                            end
                            
                            
                            
                            count_incorrect_pred = count_incorrect_pred + 1;
                            
                        case 'feedback-ali'
                            
                            %second layer
                            delta_target = -(targets(n,:) - y);
                            sig_der_2 = y.*(1-y);
                            delta_weight_second= forward_propagation_bias'* (delta_target.*(sig_der_2));
                            
                            weights_second_layer = weights_second_layer - (learning_rate*delta_weight_second);
                            
                            %first layer
                            sig_der_1 = forward_propagation_activation.*(1-forward_propagation_activation);
                            
                            % random
                             delta_weight_first_1 = random_weight*(delta_target)';
                            
                            delta_weight_first = (bias_appended_input(n,:))'*(sig_der_1.*delta_weight_first_1');
                            
                            
                           
                            
                            weights_first_layer = weights_first_layer - (learning_rate*delta_weight_first);
                            
                             switch learning
                                case 'batch'
                                    %some problems here..
                                    total_gradient_first = delta_weight_first + total_gradient_first;
                                    
                                    total_gradient_second = total_gradient_second + delta_weight_second';
                                case 'online'
                                    %second layer update
                                    weights_second_layer = weights_second_layer - (learning_rate*delta_weight_second);
                                    
                                    %first layer update
                                    weights_first_layer = weights_first_layer - (learning_rate*delta_weight_first);
                                    
                                   
                                    
                            end
                            count_incorrect_pred = count_incorrect_pred + 1;
                            
                        
                            
                    end
            end
            
            switch error_function
                case 'sum of square'
                    
                    switch out_funct
                        case 'single'
                            err = 0.5*(targets(n,1) - y).^2;
                            total_error = err + total_error;
                        case 'multi'
                            err = 0.5*sum((targets(n,:) - y).^2);
                            total_error = err + total_error;
                            
                    end
                    
                case 'cross entropy s'
                    %calculating error error
                    %cross entropy error function
                    err = -sum(sum(targets(n,1).*log(y) + (1-targets(n,1)).*log(1-y)));
                    
                    total_error = err + total_error;
                    
                    
                case 'cross entropy m'
                    err = -sum(targets(n,:).*log(y));
                    
                    total_error = err + total_error;
                    
            end
            
        end
        
    end
    
    total_pred_error = count_incorrect_pred/ndata;
    if total_pred_error == 0
        break
    end
    
    if strcmp(learning,'batch') == 1
        avg_delta_weight_first = total_gradient_first/count_incorrect_pred;
        avg_delta_weight_second = total_gradient_second/count_incorrect_pred;
        %second layer update
        weights_second_layer = weights_second_layer - (learning_rate*avg_delta_weight_second);
        
        %first layer update
        weights_first_layer = weights_first_layer - (learning_rate*avg_delta_weight_first);
        
        uw = weights_second_layer;
        uw(1,:) = [];
    end
    
    eval_error = (total_error / count_incorrect_pred);
    %fprintf('Prediction Error value: %d \n', total_pred_error);
    fprintf('cross entropy error: %d \n', eval_error);
    count_incorrect_pred = 0;
    count_error = 0;
    total_error = 0;
    
    
    batch = batch + 1;
    
    if eval_error < error_val
        break
    end
end

first_layer = weights_first_layer;
second_layer = weights_second_layer;


