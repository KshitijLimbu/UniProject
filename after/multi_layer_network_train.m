function [optimised_weights] = multi_layer_network_train(input, targets, no_of_output, net_architecture, output_function,error_function, algorithm, learning, learning_rate, error_val)

no_of_layers = size(net_architecture, 2);
no_of_weights = no_of_layers + 1;

ndata = size(input, 1);
ntargets = size(targets, 2);

if no_of_output == 1
    out_funct = 'single';
else
    out_funct = 'multi';
end


%randomise weights
for n= 1:no_of_weights
    if n == 1
        weight = (randn(size(input, 2), net_architecture(n))/sqrt(size(input, 2) + 1))';
        bias =  randn(1, net_architecture(n))/sqrt(size(input, 2) + 1);
        
        weights = [bias' weight]';
        
        temp = strcat('weight',num2str(n));
        all_weights.(temp) = weights;
        
    elseif n < (no_of_weights)
        weight = (randn(net_architecture(n-1), net_architecture(n))/sqrt(net_architecture(n-1) + 1))';
        bias =  randn(1, net_architecture(n))/sqrt(net_architecture(n-1) + 1);
        
        weights = [bias' weight]';
        
        temp = strcat('weight',num2str(n));
        all_weights.(temp) = weights;
        %weightstrcat(wordbank,num2str(1233434))
        
    else
        end_weight = (randn(net_architecture(n-1), ntargets)/sqrt(net_architecture(n-1) + 1))';
        bias = randn(1, ntargets)/sqrt(net_architecture(n-1) + 1);
        
        weights = [bias' end_weight]';
        
        temp = strcat('weight',num2str(n));
        all_weights.(temp) = weights;
    end
    
    
end

%random weights

for r=1:no_of_weights
    




    [a,b] = size(all_weights.(strcat('weight',num2str(r))));
    
    random_weight = -.5+(.5+.5)*rand(a,b);
    
    temp = strcat('weight',num2str(r));
    all_random_weights.(temp) = random_weight;

end


edata = 10;

bias_appended_input = [ones(ndata, 1), input];

count_incorrect_pred = 0;
total_pred_error = 10;
count_error = 0;

err = 10;

%feedback alignment

%random_weight = -.5+(.5+.5)*rand(size(input, 2)+1, no_hidden_neuron);

%batch training 1000 epochs
batch = 1;

epochs = 5;

total_error = 0;

while batch < epochs
    
    
    %input is in columns not rows!!!!
    %forward_propagation = input* first_layer_weight + ones(ndata, 1)* bias_weight;
    
    for n= 1:ndata
        
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
        
        %         for b= 2:(no_of_weights)
        %             temp1 = strcat('unchanged_weight',num2str(b));
        %             remove_bias = all_weights.(strcat('weight',num2str(b)));
        %             remove_bias(1,:) = [];
        %             all_unchanged_weight.(temp1) = remove_bias;
        %         end
        
        
        switch output_function
            
            case 'sigmoid'
                
                forward = all_forward_propagation_bias.((strcat('f_prop',num2str(no_of_layers))))*all_weights.(strcat('weight',num2str(no_of_weights)));
                y = 1./(1 + exp(-forward));
                
                target = targets(n);
                
                if y < 0.5
                    output = 0;
                else
                    output = 1;
                end
                
            case 'softmax'
                
                forward = all_forward_propagation_bias.((strcat('f_prop',num2str(no_of_layers))))*all_weights.(strcat('weight',num2str(no_of_weights)));
                
                temp = exp(forward);
                
                %y(y<realmin) = realmin;
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
                            
                            %last layer of weights
                            
                            delta_target = -(targets(n,:) - y);
                            sig_der_out = y*(1-y);
                            
                            sigma_1 = delta_target*(sig_der_out);
                            
                            delta_weight_last= sigma_1*all_forward_propagation_bias.((strcat('f_prop',num2str(no_of_layers))));
                            
                            temp = strcat('delta_w',num2str(no_of_weights));
                            all_delta_weights.(temp) = delta_weight_last';
                            
                            
                            
                            for a = no_of_layers:-1:1
                                %his does smethin
                                % forward_propagation_activation.*(1-forward_propagation_activation);
                                
                                counter_weight = no_of_weights-1;
                                
                                if a == no_of_layers
                                    %for multi :add this delta..
                                    weight_no_bias =  all_weights.(strcat('weight',num2str(a+1)));
                                    weight_no_bias(1,:) = [];
                                    
                                    sigma_2 = sigma_1 *weight_no_bias;
                                    
                                    fp =all_forward_propagation.((strcat('f_prop',num2str(a))));
                                    sig_der_nxt = fp.*(1-fp);
                                    
                                    
                                    delta_weight = all_forward_propagation_bias.((strcat('f_prop',num2str(a-1))))'*(sigma_2'.*sig_der_nxt);
                                    
                                    temp = strcat('delta_w',num2str(a));
                                    all_delta_weights.(temp) = delta_weight;
                                    
                                else
                                    depth = no_of_weights - a;
                                    
                                    for b=depth:-1:1
                                        if b == depth
                                            f = all_forward_propagation.((strcat('f_prop',num2str(no_of_layers))));
                                            deri_layer = f.*(1-f);
                                            error_layer = sigma_2.*deri_layer';
                                            
                                            temp = strcat('error_',num2str(b));
                                            all_hidden_layer_error.(temp) = error_layer;
                                            
                                        elseif b == 1
                                            wei = all_weights.(strcat('weight',num2str(a+1)));
                                            wei(1,:) = [];
                                            
                                            el = all_hidden_layer_error.(strcat('error_',num2str(b+1)));
                                            
                                            error_carry = wei*el;
                                            
                                            g = all_forward_propagation.((strcat('f_prop',num2str(b+1))));
                                            deri_layer = g.*(1-g);
                                            deri_layer=deri_layer';
                                            
                                            if a == 1
                                                %use input
                                                g = all_forward_propagation.((strcat('f_prop',num2str(1))));
                                                deri_layer = g.*(1-g);
                                                deri_layer=deri_layer';
                                                
                                                delta_weight = bias_appended_input(n,:)'*(error_carry.*deri_layer)';
                                                
                                                
                                                temp = strcat('delta_w',num2str(a));
                                                all_delta_weights.(temp) = delta_weight;
                                                
                                            else
                                                
                                                in = all_forward_propagation_bias.((strcat('f_prop',num2str(a-1))));
                                                
                                                g = all_forward_propagation.((strcat('f_prop',num2str(no_of_weights-depth))));
                                                deri_layer = g.*(1-g);
                                                deri_layer=deri_layer';
                                                
                                                delta_weight = in'*(error_carry.*deri_layer)';
                                                
                                                temp = strcat('delta_w',num2str(a));
                                                all_delta_weights.(temp) = delta_weight;
                                            end
                                            
                                        else
                                            
                                            
                                            %wei = all_weights.(strcat('weight',num2str(no_of_weights-b+1)));
                                            wei = all_weights.(strcat('weight',num2str(counter_weight)));
                                            
                                            wei(1,:) = [];
                                            
                                            el = all_hidden_layer_error.(strcat('error_',num2str(b+1)));
                                            
                                            error_carry = wei*el;
                                            
                                            
                                            %g = all_forward_propagation.((strcat('f_prop',num2str(no_of_weights-b))));
                                            g = all_forward_propagation.((strcat('f_prop',num2str(counter_weight-1))));
                                            deri_layer = g.*(1-g);
                                            deri_layer=deri_layer';
                                            
                                            error_layer = error_carry.*deri_layer;
                                            
                                            %here something..mite be wrong
                                            
                                            
                                            temp = strcat('error_',num2str(b));
                                            all_hidden_layer_error.(temp) = error_layer;
                                            
                                            
                                            counter_weight = counter_weight - 1;
                                        end
                                        
                                    end
                                    
                                    
                                end
                                
                            end
                            
                            
                            switch learning
                                case 'batch'
                                    total_gradient_first = delta_weight_first + total_gradient_first;
                                    total_gradient_second = total_gradient_second + delta_weight_second';
                                case 'online'
                                    
                                    for x=1:no_of_weights
                                        
                                        weight = all_weights.(strcat('weight',num2str(x)));
                                        change = all_delta_weights.(strcat('delta_w',num2str(x)));
                                        
                                        updated_weight = weight - (learning_rate*change);
                                        
                                        temp = strcat('weight',num2str(x));
                                        all_weights.(temp) = updated_weight;
                                    end
                                    
                            end
                            count_incorrect_pred = count_incorrect_pred + 1;
                            
                            
                        case 'feedback-ali'
                                                        
                            %last layer of weights
                            
                            delta_target = -(targets(n,:) - y);
                            sig_der_out = y*(1-y);
                            
                            sigma_1 = delta_target*(sig_der_out);
                            
                            delta_weight_last= sigma_1*all_forward_propagation_bias.((strcat('f_prop',num2str(no_of_layers))));
                            
                            temp = strcat('delta_w',num2str(no_of_weights));
                            all_delta_weights.(temp) = delta_weight_last';
                            
                            
                            
                            for a = no_of_layers:-1:1
                                %his does smethin
                                % forward_propagation_activation.*(1-forward_propagation_activation);
                                
                                counter_weight = no_of_weights-1;
                                
                                if a == no_of_layers
                                    %for multi :add this delta..
                                    ran_weight =  all_random_weights.(strcat('weight',num2str(a+1)));
                                    ran_weight(1,:) = [];
                                    
                                    sigma_2 = delta_target *ran_weight;
                                    
                                    fp =all_forward_propagation.((strcat('f_prop',num2str(a))));
                                    sig_der_nxt = fp.*(1-fp);
                                    
                                    
                                    delta_weight = all_forward_propagation_bias.((strcat('f_prop',num2str(a-1))))'*(sigma_2'.*sig_der_nxt);
                                    
                                    temp = strcat('delta_w',num2str(a));
                                    all_delta_weights.(temp) = delta_weight;
                                    
                                else
                                    depth = no_of_weights - a;
                                    
                                    for b=depth:-1:1
                                        if b == depth
                                            f = all_forward_propagation.((strcat('f_prop',num2str(no_of_layers))));
                                            deri_layer = f.*(1-f);
                                            error_layer = sigma_2.*deri_layer';
                                            
                                            temp = strcat('error_',num2str(b));
                                            all_hidden_layer_error.(temp) = error_layer;
                                            
                                        elseif b == 1
                                            wei = all_random_weights.(strcat('weight',num2str(a+1)));
                                            wei(1,:) = [];
                                            
                                            el = all_hidden_layer_error.(strcat('error_',num2str(b+1)));
                                            
                                            error_carry = wei*el;
                                            
                                            g = all_forward_propagation.((strcat('f_prop',num2str(b+1))));
                                            deri_layer = g.*(1-g);
                                            deri_layer=deri_layer';
                                            
                                            if a == 1
                                                %use input
                                                g = all_forward_propagation.((strcat('f_prop',num2str(1))));
                                                deri_layer = g.*(1-g);
                                                deri_layer=deri_layer';
                                                
                                                delta_weight = bias_appended_input(n,:)'*(error_carry.*deri_layer)';
                                                
                                                
                                                temp = strcat('delta_w',num2str(a));
                                                all_delta_weights.(temp) = delta_weight;
                                                
                                            else
                                                
                                                in = all_forward_propagation_bias.((strcat('f_prop',num2str(a-1))));
                                                
                                                g = all_forward_propagation.((strcat('f_prop',num2str(no_of_weights-depth))));
                                                deri_layer = g.*(1-g);
                                                deri_layer=deri_layer';
                                                
                                                delta_weight = in'*(error_carry.*deri_layer)';
                                                
                                                temp = strcat('delta_w',num2str(a));
                                                all_delta_weights.(temp) = delta_weight;
                                            end
                                            
                                        else
                                            
                                            
                                            %wei = all_weights.(strcat('weight',num2str(no_of_weights-b+1)));
                                            wei = all_weights.(strcat('weight',num2str(counter_weight)));
                                            
                                            wei(1,:) = [];
                                            
                                            el = all_hidden_layer_error.(strcat('error_',num2str(b+1)));
                                            
                                            error_carry = wei*el;
                                            
                                            
                                            %g = all_forward_propagation.((strcat('f_prop',num2str(no_of_weights-b))));
                                            g = all_forward_propagation.((strcat('f_prop',num2str(counter_weight-1))));
                                            deri_layer = g.*(1-g);
                                            deri_layer=deri_layer';
                                            
                                            error_layer = error_carry.*deri_layer;
                                            
                                            %here something..mite be wrong
                                            
                                            
                                            temp = strcat('error_',num2str(b));
                                            all_hidden_layer_error.(temp) = error_layer;
                                            
                                            
                                            counter_weight = counter_weight - 1;
                                        end
                                        
                                    end
                                    
                                    
                                end
                                
                            end
                            
                            
                            switch learning
                                case 'batch'
                                    total_gradient_first = delta_weight_first + total_gradient_first;
                                    total_gradient_second = total_gradient_second + delta_weight_second';
                                case 'online'
                                    
                                    for x=1:no_of_weights
                                        
                                        weight = all_weights.(strcat('weight',num2str(x)));
                                        change = all_delta_weights.(strcat('delta_w',num2str(x)));
                                        
                                        updated_weight = weight - (learning_rate*change);
                                        
                                        temp = strcat('weight',num2str(x));
                                        all_weights.(temp) = updated_weight;
                                    end
                                    
                            end
                            count_incorrect_pred = count_incorrect_pred + 1;
                            
                            
                    end
                case 'multi'
                    
                    switch algorithm
                        
                        case 'back-prop'
                            % back propagation
                            
                            %last layer of weights
                            
                            delta_target = -(targets(n,:) - y);
                            sig_der_out = y.*(1-y);
                            
                            sigma_1 = delta_target.*(sig_der_out);
                            
                            delta_weight_last= all_forward_propagation_bias.((strcat('f_prop',num2str(no_of_layers))))'*sigma_1;
                            
                            temp = strcat('delta_w',num2str(no_of_weights));
                            all_delta_weights.(temp) = delta_weight_last;
                            
                            
                            
                            for a = no_of_layers:-1:1
                                %his does smethin
                                % forward_propagation_activation.*(1-forward_propagation_activation);
                                
                                counter_weight = no_of_weights-1;
                                
                                if a == no_of_layers
                                    %for multi :add this delta..
                                    weight_no_bias =  all_weights.(strcat('weight',num2str(a+1)));
                                    weight_no_bias(1,:) = [];
                                    
                                    sigma_2 = weight_no_bias* sigma_1';
                                    
                                    fp =all_forward_propagation.((strcat('f_prop',num2str(a))));
                                    sig_der_nxt = fp.*(1-fp);
                                    
                                    
                                    delta_weight = all_forward_propagation_bias.((strcat('f_prop',num2str(a-1))))'*(sigma_2'.*sig_der_nxt);
                                    
                                    temp = strcat('delta_w',num2str(a));
                                    all_delta_weights.(temp) = delta_weight;
                                    
                                else
                                    depth = no_of_weights - a;
                                    
                                    for b=depth:-1:1
                                        if b == depth
                                            f = all_forward_propagation.((strcat('f_prop',num2str(no_of_layers))));
                                            deri_layer = f.*(1-f);
                                            error_layer = sigma_2.*deri_layer';
                                            
                                            temp = strcat('error_',num2str(b));
                                            all_hidden_layer_error.(temp) = error_layer;
                                            
                                        elseif b == 1
                                            wei = all_weights.(strcat('weight',num2str(a+1)));
                                            wei(1,:) = [];
                                            
                                            el = all_hidden_layer_error.(strcat('error_',num2str(b+1)));
                                            
                                            error_carry = wei*el;
                                            
                                            g = all_forward_propagation.((strcat('f_prop',num2str(b+1))));
                                            deri_layer = g.*(1-g);
                                            deri_layer=deri_layer';
                                            
                                            if a == 1
                                                %use input
                                                g = all_forward_propagation.((strcat('f_prop',num2str(1))));
                                                deri_layer = g.*(1-g);
                                                deri_layer=deri_layer';
                                                
                                                delta_weight = bias_appended_input(n,:)'*(error_carry.*deri_layer)';
                                                
                                                
                                                temp = strcat('delta_w',num2str(a));
                                                all_delta_weights.(temp) = delta_weight;
                                                
                                            else
                                                
                                                in = all_forward_propagation_bias.((strcat('f_prop',num2str(a-1))));
                                                
                                                g = all_forward_propagation.((strcat('f_prop',num2str(no_of_weights-depth))));
                                                deri_layer = g.*(1-g);
                                                deri_layer=deri_layer';
                                                
                                                delta_weight = in'*(error_carry.*deri_layer)';
                                                
                                                temp = strcat('delta_w',num2str(a));
                                                all_delta_weights.(temp) = delta_weight;
                                            end
                                            
                                        else
                                            
                                            
                                            %wei = all_weights.(strcat('weight',num2str(no_of_weights-b+1)));
                                            wei = all_weights.(strcat('weight',num2str(counter_weight)));
                                            
                                            wei(1,:) = [];
                                            
                                            el = all_hidden_layer_error.(strcat('error_',num2str(b+1)));
                                            
                                            error_carry = wei*el;
                                            
                                            
                                            %g = all_forward_propagation.((strcat('f_prop',num2str(no_of_weights-b))));
                                            g = all_forward_propagation.((strcat('f_prop',num2str(counter_weight-1))));
                                            deri_layer = g.*(1-g);
                                            deri_layer=deri_layer';
                                            
                                            error_layer = error_carry.*deri_layer;
                                            
                                            %here something..mite be wrong
                                            
                                            
                                            temp = strcat('error_',num2str(b));
                                            all_hidden_layer_error.(temp) = error_layer;
                                            
                                            
                                            counter_weight = counter_weight - 1;
                                        end
                                        
                                    end
                                    
                                    
                                end
                                
                            end
                            
                            
                            
                            switch learning
                                case 'batch'
                                    total_gradient_first = delta_weight_first + total_gradient_first;
                                    total_gradient_second = total_gradient_second + delta_weight_second';
                                case 'online'
                                    
                                    for x=1:no_of_weights
                                        
                                        weight = all_weights.(strcat('weight',num2str(x)));
                                        change = all_delta_weights.(strcat('delta_w',num2str(x)));
                                        
                                        updated_weight = weight - (learning_rate*change);
                                        
                                        temp = strcat('weight',num2str(x));
                                        all_weights.(temp) = updated_weight;
                                    end
                                    
                            end
                            count_incorrect_pred = count_incorrect_pred + 1;
                            
                        case 'feedback-ali'
                            
                            %last layer of weights
                            
                            delta_target = -(targets(n,:) - y);
                            sig_der_out = y.*(1-y);
                            
                            sigma_1 = delta_target.*(sig_der_out);
                            
                            delta_weight_last= all_forward_propagation_bias.((strcat('f_prop',num2str(no_of_layers))))'*sigma_1;
                            
                            temp = strcat('delta_w',num2str(no_of_weights));
                            all_delta_weights.(temp) = delta_weight_last;
                            
                            
                            
                            for a = no_of_layers:-1:1
                                %his does smethin
                                % forward_propagation_activation.*(1-forward_propagation_activation);
                                
                                counter_weight = no_of_weights-1;
                                
                                if a == no_of_layers
                                    %for multi :add this delta..
                                    ran_weight =  all_random_weights.(strcat('weight',num2str(a+1)));
                                    ran_weight(1,:) = [];
                                    
                                    sigma_2 = ran_weight*delta_target';
                                    
                                    fp =all_forward_propagation.((strcat('f_prop',num2str(a))));
                                    sig_der_nxt = fp.*(1-fp);
                                    
                                    
                                    delta_weight = all_forward_propagation_bias.((strcat('f_prop',num2str(a-1))))'*(sigma_2'.*sig_der_nxt);
                                    
                                    temp = strcat('delta_w',num2str(a));
                                    all_delta_weights.(temp) = delta_weight;
                                    
                                else
                                    depth = no_of_weights - a;
                                    
                                    for b=depth:-1:1
                                        if b == depth
                                            f = all_forward_propagation.((strcat('f_prop',num2str(no_of_layers))));
                                            deri_layer = f.*(1-f);
                                            error_layer = sigma_2.*deri_layer';
                                            
                                            temp = strcat('error_',num2str(b));
                                            all_hidden_layer_error.(temp) = error_layer;
                                            
                                        elseif b == 1
                                            wei = all_random_weights.(strcat('weight',num2str(a+1)));
                                            wei(1,:) = [];
                                            
                                            el = all_hidden_layer_error.(strcat('error_',num2str(b+1)));
                                            
                                            error_carry = wei*el;
                                            
                                            g = all_forward_propagation.((strcat('f_prop',num2str(b+1))));
                                            deri_layer = g.*(1-g);
                                            deri_layer=deri_layer';
                                            
                                            if a == 1
                                                %use input
                                                g = all_forward_propagation.((strcat('f_prop',num2str(1))));
                                                deri_layer = g.*(1-g);
                                                deri_layer=deri_layer';
                                                
                                                delta_weight = bias_appended_input(n,:)'*(error_carry.*deri_layer)';
                                                
                                                
                                                temp = strcat('delta_w',num2str(a));
                                                all_delta_weights.(temp) = delta_weight;
                                                
                                            else
                                                
                                                in = all_forward_propagation_bias.((strcat('f_prop',num2str(a-1))));
                                                
                                                g = all_forward_propagation.((strcat('f_prop',num2str(no_of_weights-depth))));
                                                deri_layer = g.*(1-g);
                                                deri_layer=deri_layer';
                                                
                                                delta_weight = in'*(error_carry.*deri_layer)';
                                                
                                                temp = strcat('delta_w',num2str(a));
                                                all_delta_weights.(temp) = delta_weight;
                                            end
                                            
                                        else
                                            
                                            
                                            %wei = all_weights.(strcat('weight',num2str(no_of_weights-b+1)));
                                            wei = all_weights.(strcat('weight',num2str(counter_weight)));
                                            
                                            wei(1,:) = [];
                                            
                                            el = all_hidden_layer_error.(strcat('error_',num2str(b+1)));
                                            
                                            error_carry = wei*el;
                                            
                                            
                                            %g = all_forward_propagation.((strcat('f_prop',num2str(no_of_weights-b))));
                                            g = all_forward_propagation.((strcat('f_prop',num2str(counter_weight-1))));
                                            deri_layer = g.*(1-g);
                                            deri_layer=deri_layer';
                                            
                                            error_layer = error_carry.*deri_layer;
                                            
                                            %here something..mite be wrong
                                            
                                            
                                            temp = strcat('error_',num2str(b));
                                            all_hidden_layer_error.(temp) = error_layer;
                                            
                                            
                                            counter_weight = counter_weight - 1;
                                        end
                                        
                                    end
                                    
                                    
                                end
                                
                            end
                            
                            switch learning
                                case 'batch'
                                    total_gradient_first = delta_weight_first + total_gradient_first;
                                    total_gradient_second = total_gradient_second + delta_weight_second';
                                case 'online'
                                    
                                    for x=1:no_of_weights
                                        
                                        weight = all_weights.(strcat('weight',num2str(x)));
                                        change = all_delta_weights.(strcat('delta_w',num2str(x)));
                                        
                                        updated_weight = weight - (learning_rate*change);
                                        
                                        temp = strcat('weight',num2str(x));
                                        all_weights.(temp) = updated_weight;
                                    end
                                    
                            end
                            count_incorrect_pred = count_incorrect_pred + 1;
                            
                            

                    end
            end
            
            switch error_function
                case 'sum of square'
                    
                    switch out_funct
                        case 'single'
                            err = 0.5*(targets(n) - y).^2;
                            
                        case 'multi'
                            err = 0.5*sum((targets(n) - y).^2);
                            
                            fprintf('error: %d \n', err);
                    end
                    
                case 'cross entropy s'
                    %calculating error error
                    %cross entropy error function
                    err = -sum(sum(targets(n).*log(y) + (1-targets(n)).*log(1-y)));
                    total_error = err + total_error;
                    
                case 'cross entropy m'
                    err = -sum(targets(n,:).*log(y));
                    
                    total_error = err + total_error;
            end
            
        end
        
        
        
        %end of the whole loop for data
    end
    
    
    
    total_pred_error = count_incorrect_pred/ndata;
    if total_pred_error == 0
        break
    end
    eval_error = (total_error / count_incorrect_pred);
    %fprintf('Prediction Error value: %d \n', total_pred_error);
    fprintf('cross entropy error: %d \n', eval_error);
    
    %fprintf('error: %d \n', count_error);
    count_incorrect_pred = 0;
    count_error = 0;
    
    total_error = 0;
    batch = batch + 1;
    
    if eval_error < error_val
        break
    end
    
end

optimised_weights = all_weights;



