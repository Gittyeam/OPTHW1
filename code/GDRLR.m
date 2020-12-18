
function [w,wVec,it,loss,ttot,lossVec,timeVec,gnrit,err] = GDRLR(X,y,w,reg,L,maxit,rate)

%------------------------------------------------
% Gradient Descent method with fixed step for RLR
%------------------------------------------------

% INPUT
% X: matrix of sizes (m,n), m istances of dimension n
% y: col vector of length m, it contains the corresponding label for each
% istance in X (binary classification -1/+1)
% w: row vector of length n, starting values for parameters
% reg: scalar, regularization term
% L: Lipschitz constant of the gradient
% maxit: maximum number of iterations
% rate: weight vector computing rate

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

n = size(X,2);

% initialize vectors of loss, grad norm and time
lossVec = zeros(1,maxit);
gnrit = zeros(1,maxit);
timeVec = zeros(1,maxit);
wVec = zeros(maxit/rate,n);
wVec(1,:) = w;

loss = LossRLR(X,y,w,reg);
it = 1;
nr = 1;
err = 0;

% start time
tic;

% iterate until max number of iteration reached
while (it<=maxit)
    % vectors updating
    if (it > 1)
        timeVec(it) = toc;
    else
        timeVec(it) = 0;
    end
    
    lossVec(it) = loss;
    
    % gradient evaluation
    g = GradLossRLR(X,y,w,reg);
    
    % check if the 
    if isnan(g)
        disp('Gradient overflow');
        err = 1;
        break;
    end
    
    % compute direction and gradient norm
    d = -g;
    gnr = g*d';
    gnrit(it) = -gnr;
    
    % linesearch with fixed step alpha
    alpha = 1/L;
    w = w+alpha*d;
    loss = LossRLR(X,y,w,reg);
    
    if(mod(it,rate)==0)
        nr = nr+1;
        wVec(nr,:) = w;
    end
                         
    it = it+1;
end

ttot = toc;
it = it-1;

end
