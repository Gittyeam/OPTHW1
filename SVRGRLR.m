function [w,wVec,it,loss,ttot,lossVec,timeVec,gnrit,err] = SVRGRLR(X,y,w,reg,lc,...
    verbosity,nepochs,maxit)

%------------------------------------------------------------------
%Stochastic Variance Reduction Gradient Method for Regularized Logistic Regression
%------------------------------------------------------------------

%INPUT 
%X: matrix of sizes (m,n), m istances of dimension n
%y: col vector of length m, it contains the corresponding label for each
%istance in X (binary classification -1/+1)
%w: row vector of length n, starting values for parameters
%reg: scalar, regularization term
%lc: constant of the reduced stepsize (numerator)
%verbosity: printing level
%nepochs: epoch length
%maxit: maximum number of iterations

%OUTPUT
%w: optimal vector of parameters
%wVec: matrix with all the obtained vector w
%it: number of performed iterations
%loss: value of the loss function
%ttot: total CPU time of execution
%lossVec: vector containing the loss value for each iter
%timeVec: vector containing CPU time at each iter
%gnrit: vector containing the norm of the gradient at each iter
%err: error flag for overflow
%------------------------------------------------------------------

%initialize vectors of loss, grad norm and time

lossVec = zeros(1,maxit);
gnrit = zeros(1,maxit);
timeVec = zeros(1,maxit);

%Start time
tic;

%Data dimensions
m = size(X,1);

gsvrg = GradLossRLR(X,y,w,reg);     % vector containing the SUM of gradients (m * \mu tilde)
loss = LossRLR(X,y,w,reg);          % objective function computation
wz = w;                             % w tilde for the first epoch

it = 1;
err = 0;


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
    ei = exp(-yi*xi*w.');
    eiz = exp(-yi*xi*wz.');
    g = -yi*ei/(1+ei) * xi + reg*w/m;       % new ind-th gradient for iteration it
    gz = -yi*eiz/(1+eiz) * xi + reg*wz/m;	% epoch ind-th gradient
    gf = g-gz+(1/m)*gsvrg;                  % complete ind-th gradient for iteration it
    
    % check gradient overflow
    if isnan(gf)
        disp('Gradient overflow');
        err=1;
        break;
    end
    
    d = -gf;
    gnr = gf*gf';
    gnrit(it) = gnr;
    
    %alpha selection
    alpha = lc;
    z = w+alpha*d;
    % update w tilde, loss and gradient at the end of epoch (nepochs iterations per epoch)
    if(mod(it,nepochs)==0)
        wz = z;
        hz = LossRLR(X,y,z,reg);
        gsvrg = GradLossRLR(X,y,z,reg);
    else
        hz=loss;
    end
    
    w=z;
    loss = hz;
    
    if(it==1)
        wVec=w;
    end
    
    if((it>1) && (mod(it-1,1000)==0))
        wVec(size(wVec,1)+1,:)=w;
    end
    
    if (verbosity>0)
        disp(['-----------------** ' num2str(it) ' **------------------']);
        disp(['gnr      = ' num2str(gnr)]);
        disp(['h(w)     = ' num2str(loss)]);
        disp(['alpha     = ' num2str(alpha)]);
    end
    
    it = it+1;

end

ttot = toc;

it=it-1;
end
