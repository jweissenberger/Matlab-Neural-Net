function [ Yout, z4, z3, z2 ] = ForwardProp( X, W12, b12, W23, b23, W34, b34 )
%Forward Propagation algorithm 
%   This computest the forward propagation for this particular neural net.
%   Inputs: input vector X
%           Weights: W12, W23, W34
%   Output: prediction vector Yout


%% Input layer to First Hidden Layer
z2 = (X*W12);% activity going into the second layer
[m,~] = size(z2);
z2 = z2 + ones(m, 1) * b12; % adding the basis to all of the examples


%% First Hidden Layer to Second Hidden Layer
a2 = sigmoid(z2);
z3 = (a2*W23);
[m,~] = size(z3);
z3 = z3 + ones(m, 1) * b23;


%% Second Hidden Layer to Output
a3 = sigmoid(z3);
z4 = (a3*W34);
[m,~] = size(z4);
z4 = z4 + ones(m, 1) * b34;
Yout = sigmoid(z4);


end

