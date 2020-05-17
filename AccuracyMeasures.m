
function [precision,recall,F1,accuracy] = AccuracyMeasures(y_pred,y_true)

%------------------------------------------------------------------
%Compute Precision, Recall, F1 and accuracy score for binary 
%classification problem (+1/-1)
%------------------------------------------------------------------

%INPUT
%y_pred: vector (m,1) of predicted labels
%y_true: vector (m,1) of true labels

%OUTPUT
%precision = TP /(TP+FP)
%recall = TP / (TP+FN)
%F1 = 2*(precision*recall)/(precision+recall)
%accuracy = % of correctly classified
%------------------------------------------------------------------

%compute true/false positive/negative
TP = 0;
FP = 0;
TN = 0;
FN = 0;
for i = 1:size(y_pred)
    if (y_pred(i) >= 0 && y_true(i) >= 0)
        TP = TP + 1;
    end
    if (y_pred(i) >= 0 && y_true(i) < 0)
        FP = FP + 1;
    end
    if (y_pred(i) < 0 && y_true(i) >= 0)
        FN = FN + 1;
    end
    if (y_pred(i) < 0 && y_true(i) < 0)
        TN = TN + 1;
    end
end
    
%compute accuracy measures
precision = TP /(TP+FP);
recall = TP / (TP+FN);
F1 = 2*(precision*recall)/(precision+recall);

%accuracy
accuracy = (TP+TN)/(TP+TN+FN+FP);

end