function [w,wVec,it,loss,ttot,lossVec,timeVec,gnrit,err] = SGRLR(X,y,w,reg,LC,...
          maxit,rate)

%---------------------------------------------------------------
% Stochastic Gradient Method for Regularized Logistic Regression
%---------------------------------------------------------------

% INPUT 
% X: matrix of sizes (m,n), m istances of dimension n
% y: col vector of length m, it contains the corresponding label for each
% istance in X (binary classification -1/+1)
% w: row vector of length n, starting values for parameters
% reg: scalar, regularization term
% LC: constant of the reduced stepsize (numerator) 
% maxit: maximum number of iterations
% rate: loss and weight vector computing rate

% OUTPUT
% w: optimal vector of parameters
% wVec: matrix with all the obtained vector w
% it: number of performed iterations
% loss: value of the loss function
% ttot: total CPU time of execution
% lossVec: vector containing the loss value for each iter
% timeVec: vector containing CPU time at each iter
% gnrit: vector containing the norm of the gradient at each iter
% err: error flag for overflow
%------------------------------------------------------------------

% Data dimensions
[m,n] = size(X);

% initialize vectors of loss, grad norm and time
lossVec = zeros(1,maxit);
gnrit = zeros(1,maxit);
timeVec = zeros(1,maxit);
wVec = zeros(maxit/rate,n);
wVec(1,:) = w;

% Loss function computation
loss = LossRLR(X,y,w,reg);

it = 1;
nr = 1;
err = 0;

% start time
tic;

while (it<=maxit)
    %vectors updating
    if (it > 1)
        timeVec(it) = toc;
    else
        timeVec(it) = 0;
    end
    
    lossVec(it)=loss;
    
    % gradient evaluation
    ind = randi(m);
    xi = X(ind,:);
    yi = y(ind);
    g = GradLossRLR(xi,yi,w,reg/m);
    
    % check gradient overflow
    if isnan(g)
        disp('Gradient overflow');
        err = 1;
        break;
    end
    
    %compute direction
    d = -g;
    
    %reduced alpha
    alpha = sqrt(LC/(it+1));        % test without sqrt
    
    w = w+alpha*d;
    
    if(mod(it,rate)==0)
        nr = nr+1;
        loss = LossRLR(X,y,w,reg);
        wVec(nr,:) = w;
    end
    
    it = it+1;
end

ttot = toc;
it = it-1;

end
