function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. 

g = zeros(size(z));

g = (0.15*exp(-0.15*z))./(exp(-0.15*z)+1).^2;



% =============================================================




end
