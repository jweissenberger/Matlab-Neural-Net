function [ del4, del3, del2 ] = Backprop( Yprediction, Yactual, z4, z3, z2, W34, W23 )
%Backprop the back propagation algorithm
%   Detailed explanation goes here

% error in the output layer
del4 = Yprediction - Yactual.*sigmoidGradient(z4);

% error in the third layer (second hidden layer)
del3 = (W34'*del4).*sigmoidGradient(z3);

% error in the second layer (first hidden layer)
del2 = (W23'*del3).*sigmoidGradient(z2);

end

