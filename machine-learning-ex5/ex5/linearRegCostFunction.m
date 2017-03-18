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

n = size(theta);

pred = X * theta;
error = pred - y;
J1 = (error) .* (error);
J1 = sum(J1(:)) / (2 * m);
J2 = theta .* theta;
J2 = lambda * sum(J2(2:n)) / (2 * m);
J = J1 + J2;

n = size(theta, 1);

for i=1:n
  temp = transpose(error) * X(:,i);
  temp = sum(temp);
  grad(i:i,:) = temp / m;
endfor

for i=2:n
  temp = lambda * theta(i:i,:) / m;
  grad(i:i,:) += temp;
endfor
% =========================================================================

grad = grad(:);

end
