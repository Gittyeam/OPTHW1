function [w,wVec,it,loss,ttot,lossVec,timeVec,gnrit,err] = SGRLR(X,y,w,reg,LC,...
    maxit,verbosity)

%------------------------------------------------------------------
% Stochastic Gradient Method for Regularized Logistic Regression
%------------------------------------------------------------------

%INPUT 
%X: matrix of sizes (m,n), m istances of dimension n
%y: col vector of length m, it contains the corresponding label for each
%istance in X (binary classification -1/+1)
%w: row vector of length n, starting values for parameters
%reg: scalar, regularization term
%LC: constant of the reduced stepsize (numerator) 
%maxit: maximum number of iterations
%verbosity: printing level

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

lossVec=zeros(1,maxit);
gnrit=zeros(1,maxit);
timeVec=zeros(1,maxit);

%Start time
tic;

%Data dimensions
[m,n] = size(X);

%Loss function computation
loss=LossRLR(X,y,w,reg);

it=1;
err = 0;

while (it<=maxit)
    %vectors updating
    if (it==1)
        timeVec(it) = 0;
    else
        timeVec(it) = toc;
    end
    lossVec(it)=loss;
    
    % gradient evaluation
    ind=randi(m);
    xi=X(ind,:);
    yi=y(ind);
    ei=exp(-yi*xi*w');
    g=(-ei*yi/(1+ei))*xi+(reg/m)*w;
    
    % check gradient overflow
    if isnan(g)
        disp('Gradient overflow');
        err=1;
        break;
    end
    
    %compute direction and gradient norm
    d=-g;  
    gnr = g*d';
    gnrit(it) = -gnr;
       
    %reduced alpha
    alpha =sqrt(LC/(it+1));
    
    new_w = w+alpha*d;
    
    if(mod(it,1000)==0)
        new_loss = LossRLR(X,y,new_w,reg);
    else
        new_loss = loss;
    end
        
    w=new_w; 
    loss = new_loss;
    
    if(it==1)
        wVec=w;
    end
    
    if((it>1)&& (mod(it-1,100)==0))
        wVec(size(wVec,1)+1,:)=w;
    end

        
    if (verbosity>0)
        disp(['-----------------** ' num2str(it) ' **------------------']);
        disp(['gnr      = ' num2str(abs(gnr))]);
        disp(['loss     = ' num2str(loss)]);
        disp(['alpha    = ' num2str(alpha)]);                    
    end

    it = it+1;
        
        
end

ttot = toc;
if(it<=maxit)
    lossVec=lossVec(1:it-1);
    timeVec=timeVec(1:it-1);
    gnrit=gnrit(1:it-1);
end

it=it-1;
end

