# Optimization for Data Science - Homework 1

**Students:**\
Caria Natascia       1225874\
Cozzolino Claudia    1227998\
Petrella Alfredo     1206627

## Task
Regularized Logistic Regression (RLR) with
- Gradient Descent (fixed step size)
- Stochastic Gradient Descent
- Stochastic Variance Reduced Gradient Descent

## Data Set
GISETTE Data Set for handwritten digit recognition problem.
The task is to separate the highly confusible digits '4' and '9'.\
It consists of 6000 instances, 5000 attributes and 2 classes.\
The Data is available on [UCI archive](https://archive.ics.uci.edu/ml/datasets/Gisette).

## Code Description
| File Name          |       Description                                              |
|--------------------|----------------------------------------------------------------|
| GDRLR.m            | Descent method with fixed step size for RLR                    |
| SGRLR.m            | Stochastic Gradient Descent method for RLR                     |
| SVRGRLR.m          | Stochastic Variance Reduced Gradient Descent method for RLR    |
| LossRLR.m          | Regularized Logistic Loss Function computation                 |
| GradLossRLR.m      | Regularized Logistic Loss Function full Gradient computation   |
| AccuracyMeasures.m | Precision, Recall, F1 and Accuracy score computation           |
| PrintResults.m     | Print method results in terms of time, iterations and accuracy |
| MainHW1.m          | Parameters initialization and GM, SGM, SVRGM comparison        |
