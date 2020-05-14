%HOMEWORK 1 - OPTIMIZATION FOR DATA SCIENCE - 10/5/2020

%Caria Natascia
%Cozzolino Claudia
%Petrella Alfredo

%IMPORT DATASET AND PREPROCESS

datagis = load('data.mat');
data=datagis.gis;

% Train-test split
X_train=[data.Xtrain ones(size(data.Xtrain,1),1)];
X_test= [data.Xtest ones(size(data.Xtest,1),1)];
y_train = data.ytrain;
y_test = data.ytest;

[m, n] = size(X_train);

%REGULARIZED LOGISTIC REGRESSION

%Initilize hyper-parameters

%initialize parameters to Std Normal random values
w_gm=randn(1,n);
w_sg=randn(1,n);
w_svrg=randn(1,n);

%min-max normalization in [-1 , 1]
w_gm = w_gm/max(abs(w_gm));  
w_sg = w_sg/max(abs(w_sg));
w_svrg = w_svrg/max(abs(w_svrg));

%initilize hyper-parameters
reg_gm=0.01;
reg_sg=10;
reg_svrg=0.01;

%Lipschitz constants extimation
L=10^6;
LC=0.001;

%maximum number of iterations (use multiple of 100 for the print function)
maxit_gm=100;
maxit_sg=25000;
maxit_svrg = 25000;

%tolerance for the stopping condition on the gradient norm
eps_gm=1.0e-1;
eps_sg=1.0e-3;
eps_svrg=1.0e-2;

%output type
verbosity=0;

%class all open figures (if any)
close all

%1) GRADIENT DESCENT FIXED STEPSIZE

%call GD_rlr
disp('*****************************');
disp('*        GM STANDARD        *');
disp('*****************************');
[optw_gm,wVec_gm,it_gm,loss_gm,ttot_gm,lossVec_gm,timeVec_gm,gnrit_gm,err_gm] = ...
GDRLR(X_train,y_train,w_gm,reg_gm,L,maxit_gm,eps_gm,verbosity);

% Print results:
PrintResults('GM',X_train,X_test,y_train,y_test,optw_gm,wVec_gm,it_gm,loss_gm,ttot_gm,lossVec_gm,timeVec_gm,gnrit_gm,err_gm)

%2) STOCHASTIC GRADIENT DESCENT

disp('*****************************');
disp('*        SGM STANDARD       *');
disp('*****************************');

%call STG_rlr
[optw_sg,wVec_sg,it_sg,loss_sg,ttot_sg,lossVec_sg,timeVec_sg,gnrit_sg,err_sg] = ...
SGRLR(X_train,y_train,w_sg,reg_sg,LC,maxit_sg,eps_sg,verbosity);

% Print results:
PrintResults('SGM',X_train,X_test,y_train,y_test,optw_sg,wVec_sg,it_sg,loss_sg,ttot_sg,lossVec_sg,timeVec_sg,gnrit_sg,err_sg)

%3) SVRG

disp('*****************************');
disp('*       SVRGM STANDARD      *');
disp('*****************************');

%call SVRG_rlr
nepochs=1000;
fstop = 1e-1; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% check it (- to ignore it)
stopcr = 1; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% check it


[optw_svrg,wVec_svrg,it_svrg,loss_svrg,ttot_svrg,lossVec_svrg,timeVec_svrg,gnrit_svrg,err_svrg] = SVRGRLR(X_train,y_train,w_svrg,reg_svrg,LC,...
    verbosity,nepochs,maxit_svrg,eps_svrg,fstop,stopcr);

% Print results:
PrintResults('SVRG',X_train,X_test,y_train,y_test,optw_svrg,wVec_svrg,it_svrg,loss_svrg,ttot_svrg,lossVec_svrg,timeVec_svrg,gnrit_svrg,err_svrg)

%COMPARE METHODS
if(err_gm+err_sg+err_svrg==0)
    %plot time - loss
    figure
    semilogy(timeVec_gm,lossVec_gm,'r-')
    hold on
    semilogy(timeVec_sg,lossVec_sg,'b-')
    semilogy(timeVec_sg,lossVec_svrg,'g-')
    xlim([0 125]);
    xlabel('Time'); 
    ylabel('Loss');
    title('GD vs SGD vs SVRGD - Loss function')
    legend('GM', 'SGM', 'SVRGM')
    
    %log log time plot time - loss 
    figure
    loglog(timeVec_gm,lossVec_gm,'r-')
    hold on
    loglog(timeVec_sg,lossVec_sg,'b-')
    loglog(timeVec_sg,lossVec_svrg,'g-')
    xlabel('Time'); 
    ylabel('Loss');
    %xlim([0 1000]);
    title('GD vs SGD vs SVRGD - Loss function')
    legend('GM', 'SGM', 'SVRGM') 
    
end   
   
 
