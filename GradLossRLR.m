
function [grad]=GradLossRLR(X,y,w,reg)

%------------------------------------------------------------------
%RLR Gradient vector
%------------------------------------------------------------------

%INPUT
%X: matrix of sizes (m,n), m istances of dimension n
%y: col vector of length m, it contains the corresponding label for each
%istance in X (binary classification -1/+1)
%w: row vector of length n, parameter
%reg: scalar, regularization term

%OUTPUT
%grad: vector of gradient (1,n)

grad=zeros(1,size(X,2));
for i=1:size(X,1)
    x=X(i,:);
    eyxw=exp(-y(i)*x*w');
    grad=grad+(-eyxw*y(i)/(1+eyxw))*x;
end
grad=grad+reg*w;
end