function [data targetVec] = GenerateGaussianData(samples)
% Generates a two-class dataset in matrix 'data' with 2D data vectors and 
% also indicates each pattern class in the matrix 'targetVec'

% Function first generates 'samples' 2-d data vectors of class 1 by independently 
% drawing each dimension from a  Gaussian distribution.
% class 1 patterns have an associated output label vector in matrix targetVec [1 0]'  
% The  mean and covariance for class 1 are
% Mean1 = [4; -1;];
% Sigma1 = [0.6 .8; .8 7];

% Function then generates 'samples' 2-d data vectors of class 2 by independently 
% drawing each dimension from a  Gaussian distribution. 
% class 2 patterns have an associated output label vector in matrix targetVec[0 1]'
% The  mean and covariance for class 2 are
% Mean2 = [-1; 1;];
% Sigma2 = [0.7 .8; .8 9];

% the two sets of class1 and class2 data are then concatenated into the
% output  matrix 'data'
% similarly the label vectors are concatenated onto the output matrix
% 'targetVec'

% Both output matrices therefore have size = (2, 2 * samples)

% see example usage at end of function
% set demo = 1 to run this plot example
demo = 0; % demo off
%demo = 1; % demo on

% Ian Howard, University of Plymouth, December 2016
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% set mean and covariance for class 0
Mean1 = [4; -1;];
Sigma1 = [0.6 .8; .8 7];

% set mean and covariance for class 1
Mean2 = [-1; 1;];
Sigma2 = [0.7 .8; .8 9];


% generate Gaussian distibuted dataset for class 0
data1 =  chol(Sigma1) * randn(2, samples) + repmat(Mean1,1,samples);
% generate target vectors for class 0
label1 = repmat([1; 0;] , 1, samples);
disp(cov(data1'));

% generate Gaussian distibuted datasetn for class 1
data2 =  chol(Sigma2) * randn(2, samples) + repmat(Mean2,1,samples);
% generate target vectors for class 1
label2 = repmat([0; 1;] , 1, samples);
disp(cov(data2'));

% concatenate class0 and class1 datasets into single dataset
data = [data1 data2];
targetVec = [label1 label2];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(demo)
    % example usage
    % vectorized example to extract all class 1 patterns
    % examine first dimension which is 1 for class 1
    % you could also use a loop if you wish
    fidx = find(targetVec(1,:) == 1);
    c1data = data(:,fidx);
    
    % vectorized example to extract all class 2 patterns
    % examine first dimension which is 0 for class 2
    % you could also use a loop if you wish
    fidx = find(targetVec(1,:) == 0);
    c2data = data(:,fidx);
    
    % now plot separated classes  on a figure
    figure
    hold on
    plot(c1data(1,:), c1data(2,:), 'ro');
    plot(c2data(1,:), c2data(2,:), 'b+');
    xlabel('x-dimension');
    ylabel('y-dimension');
    title('Ploting class1 patterns in red and class2 patterns in blue');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





