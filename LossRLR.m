
function [loss]=LossRLR(X,y,w,reg)

%------------------------------------------------------------------------
% RLR LOSS FUNCTION
%------------------------------------------------------------------------

% INPUT
% X: matrix of sizes (m,n), m istances of dimension n
% y: col vector of length m, it contains the corresponding label for each
% istance in X (binary classification -1/+1)
% w: row vector of length n, parameter
% reg: scalar, regularization term

% OUTPUT
% loss: loss function value (scalar)
%------------------------------------------------------------------------

loss=0;
for i=1:size(X,1)
    x=X(i,:);
    h=x*w';
    l=log(1+exp(-y(i)*h));
    loss=loss+l;
end
loss=loss+(reg/2)*(w*w');

end
