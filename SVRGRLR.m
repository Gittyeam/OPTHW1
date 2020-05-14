%Stochastic Variance Reduction Gradient Method for Regularized Logistic Regression

function [w,wVec,it,hw,ttot,hwVec,timeVec,gnrit,err] = SVRGRLR(X,y,w,reg,lc,...
    verbosity,nepochs,maxit,eps,fstop,stopcr)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Implementation of the Stochastic Variance Reduction 
% Gradient Method for Regularized Logistic Regression
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% for min f(x)=0.5 \|Qx-c\|^2
%
%INPUTS:
% X: data matrix of sizes (m,n), m istances of dimension n
% y: label column vector of length m (binary classification -1/+1)
% w: starting point
% reg: regularization term
% lc: constant of the reduced stepsize (numerator)
% verbosity: printing level
% nepochs: epoch length
% maxit: maximum number of iterations
% eps: tolerance
% fstop: target o.f. value
% stopcr: stopping condition
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%OUTPUTS:
% w: optimal vector of parameters
% it: number of performed iterations
% hw: value of the loss function
% ttot: total CPU time of execution
% hwVec: values of the loss function at each iteration
% timeVec: CPU time at each iter
% gnrit: squared norms of the gradient at each iter
% err: error flag for overflow


hwVec=zeros(1,maxit);
gnrit=zeros(1,maxit);
timeVec=zeros(1,maxit);

flagls=0; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic;
timeVec(1) = 0;

m = size(X,1);

gsvrg = GradLossRLR(X,y,w,reg);  % vector containing the SUM of gradients (m * \mu tilde)
hw = LossRLR(X,y,w,reg);            % objective function computation
wz = w;                             % w tilde for the first epoch

it=1;
timeVec(it) = 0;
err = 0; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% jmvlsdjefkwrfreg

while (flagls==0)
    %vectors updating
    if (it>1)
        timeVec(it) = toc;
    end
    hwVec(it)=hw; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % not here! or maybe yes?
    
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
    % gnr = g*d'; % ???????????????????????????????????????????????????????
    % gnrit(it) = -gnr; % ?????????????????????????????????????????????????
    gnr = gf*gf'; % ?????????????????????????????????????????????????????
    gnrit(it) = gnr; % ??????????????????????????????????????????????????
    
    % stopping criteria and test for termination
    if (it>=maxit)
        break;
    end
        switch stopcr %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            case 1
                % continue if not yet reached target value fstop
                if (hw<=fstop)
                    break
                end
            case 2
                % stopping criterion based on the product of the 
                % gradient with the direction
                if (gnr <= eps)
                    break;
                end
            otherwise
                error('Unknown stopping criterion');
        end % end of the stopping criteria switch
        
        %alpha selection
        alpha = lc;
        z = w+alpha*d;
        % update w tilde, loss and gradient at the end of epoch (nepochs iterations per epoch)
        if(mod(it,nepochs)==0)
            wz = z;
            hz = LossRLR(X,y,z,reg);
            gsvrg = GradLossRLR(X,y,z,reg);
        else
            hz=hw;
        end
        
        w=z;
        hw = hz;
        if(it==1)
            wVec=w;
        end
        if(mod(it-1,100)==0)
            wVec(size(wVec,1)+1,:)=w;
        end
        if (verbosity>0)
            disp(['-----------------** ' num2str(it) ' **------------------']);
            disp(['gnr      = ' num2str(gnr)]);
            disp(['h(w)     = ' num2str(hw)]);
            disp(['alpha     = ' num2str(alpha)]);                    
        end
        
        it = it+1;
        
        
end

if(it<maxit)
    hwVec(it+1:maxit)=hwVec(it);
    gnrit(it+1:maxit)=gnrit(it);
    timeVec(it+1:maxit)=timeVec(it);
end

ttot = toc;


end
