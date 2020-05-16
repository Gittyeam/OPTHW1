function [w,wVec,it,loss,ttot,lossVec,timeVec,gnrit,err] = SVRGRLR(X,y,w,reg,alpha,...
          epochs_len,maxit)

%----------------------------------------------------------------------------------
% Stochastic Variance Reduction Gradient Method for Regularized Logistic Regression
%----------------------------------------------------------------------------------

% INPUT 
% X: matrix of sizes (m,n), m istances of dimension n
% y: col vector of length m, it contains the corresponding label for each
% instance in X (binary classification -1/+1)
% w: row vector of length n, starting values for parameters
% reg: scalar, regularization term
% alpha: fixed stepsize
% epochs_len: epochs length
% maxit: maximum number of iterations

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
[m, n] = size(X);

% initialize vectors of loss, grad norm and time
lossVec = zeros(1,maxit);
gnrit = zeros(1,maxit);
timeVec = zeros(1,maxit);
wVec = zeros(maxit/epochs_len,n);      % Memory consuming, but wonderful charts
wVec(1,:) = w;

gsvrg = GradLossRLR(X,y,w,reg);     % vector containing the SUM of gradients (m * \mu tilde)
gnr = gsvrg*gsvrg';
loss = LossRLR(X,y,w,reg);          % objective function computation
wz = w;                             % w tilde for the first epoch

it = 1;
ep = 1;
err = 0;

% start time
tic;

while (it<=maxit)
    % vectors updating
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
    g = GradLossRLR(xi,yi,w,reg/m);         % new ind-th gradient for iteration it
    gz = GradLossRLR(xi,yi,wz,reg/m);       % epoch ind-th gradient
    
    gf = g-gz+(1/m)*gsvrg;                  % complete ind-th gradient for iteration it
    
    % check gradient overflow
    if isnan(gf)
        disp('Gradient overflow');
        err = 1;
        break;
    end
    
    d = -gf;
    gnrit(it) = gnr;
    
    % weight update
    w = w+alpha*d;
    
    % update w tilde, loss and gradient at the end of epoch (epochs_len iterations per epoch)
    if(mod(it,epochs_len)==0)
        ep = ep+1;
        wz = w;
        wVec(ep,:) = wz;
        loss = LossRLR(X,y,wz,reg);
        gsvrg = GradLossRLR(X,y,wz,reg);
        gnr = gsvrg*gsvrg';
    end
    
    it = it+1;
end

ttot = toc;
it=it-1;

end
