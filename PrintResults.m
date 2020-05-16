function [t,accVec,F1Vec]=PrintResults(name,X_train,X_test,y_train,y_test,optw,wVec,it,loss,ttot,lossVec,timeVec,rate,err,gnrit)

%------------------------------------------------------------------
%This function print the results of a method in terms of time, iterations
%and accuracy
%------------------------------------------------------------------

if(err==0)
    fprintf(1,strcat(name,' Loss            = %10.3e\n'),loss);
    fprintf(1,strcat(name,' Iterations      = %d\n'),it);
    if (name != 'SGM')
       fprintf(1,strcat(name,' ||gr||^2        = %10.3e\n'),gnrit(it));
    end
    fprintf(1,strcat(name,'  CPU time       = %10.3e\n'), ttot);
    
    %plot loss as function of iter and time 
    figure('Name',strcat('1 - ',name))
    title('Loss - Iter');
    semilogy(1:it,lossVec,'k-')
    title(strcat(name,' - Loss function'))
    xlabel('Iter'); 
    ylabel('Loss');

    figure('Name',strcat('2 - ',name))
    title('Loss - Time');
    semilogy(timeVec,lossVec,'r-')
    title(strcat(name,' - Loss function'))
    xlabel('Time'); 
    ylabel('Loss');
    
    %plot grad norm 
    if (name != 'SGM')
        figure('Name',strcat('3 - ',name))
        title('Gradient - Iter')
        semilogy(1:it,gnrit,'g-')
        title(strcat(name,' - Gradient Norm ^2'))
        xlabel('Iter'); 
        ylabel('||gr||^2');
    end
    
    %Train accuracy
    
    F1Vec=zeros(1,size(wVec,1));
    accVec=zeros(1,size(wVec,1));
    
    for i=1:size(wVec,1)
        y_pred=sign(X_train*wVec(i,:)');
        [prec,rec,F1,acc] = AccuracyMeasures(y_pred,y_train);
        F1Vec(i)=F1;
        accVec(i)=acc;
    end
    
    
    % Print evaluations:
    fprintf(1,strcat(name,' train precision = %4.2f\n'),prec);
    fprintf(1,strcat(name,' train recall    = %4.2f\n'),rec);
    fprintf(1,strcat(name,' train F1        = %4.2f\n'),F1);
    fprintf(1,strcat(name,' train accuracy  = %4.2f\n'),acc);
    
    t=[timeVec(1:rate:it) timeVec(it)];
    
    figure('Name',strcat('4 - ',name))
    plot(t,accVec,'b-')
    title(strcat(name,' - Train Accuracy'))
    xlabel('time'); 
    ylabel('accuracy %');

    figure('Name',strcat('5 - ',name))
    plot(t,F1Vec,'b-')
    title(strcat(name,' - F1 score'))
    xlabel('time'); 
    ylabel('F1 score');

    
    %Test accuracy
    
    %predict
    y_pred=sign(X_test*optw');
    
    %compute accuracy scores
    [prec,rec,F1,acc] = AccuracyMeasures(y_pred,y_test);

    % Print evaluations:
    fprintf(1,strcat(name,' test precision  = %4.2f\n'),prec);
    fprintf(1,strcat(name,' test recall     = %4.2f\n'),rec);
    fprintf(1,strcat(name,' test F1         = %4.2f\n'),F1);
    fprintf(1,strcat(name,' test accuracy   = %4.2f\n'),acc);
else
    fprintf('The method does not converge! \n')
end

end 
