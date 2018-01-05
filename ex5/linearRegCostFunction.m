function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h = X*theta;
cost = (h - y);
t = theta(2:end);
J = (1/m/2) * (cost'*cost) + (lambda/m/2) * (t' * t);

grad = (1/m) * ((h-y)' * X) + (lambda/m) * theta';
grad(1) = (1/m) * ((h-y)' * X(:, 1));


% =========================================================================

grad = grad(:);

end
