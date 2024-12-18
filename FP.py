def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix, accuracy_score
import sklearn.metrics as metrics

df = pd.read_csv("Weather_Data.csv")
#print(df.head(5))

''' DATA PROCESSING '''

#First, we need to perform one hot encoding to convert categorical variables to binary variables.
df_sydney_processed = pd.get_dummies(data=df, columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])

#Next, we replace the values of the 'RainTomorrow' column changing them from a categorical column to a binary column. We do not use the `get_dummies` method because we would end up with two columns for 'RainTomorrow' and we do not want, since 'RainTomorrow' is our target.
df_sydney_processed.replace(['No', 'Yes'], [0,1], inplace=True)

'''Training Data and Test Data'''
#Now, we set our 'features' or x values and our Y or target variable.
df_sydney_processed.drop('Date',axis=1,inplace=True)
df_sydney_processed = df_sydney_processed.astype(float)
#print(df_sydney_processed.head(5))

features = df_sydney_processed.drop(columns='RainTomorrow', axis=1)
Y = df_sydney_processed['RainTomorrow']
#print(features)
#print(Y)

''' Linear Regression '''
# Q1) Use the `train_test_split` function to split the `features` and `Y` dataframes with a `test_size` of `0.2` 
# and the `random_state` set to `10`.
features_train, features_test, Y_train, Y_test = train_test_split(features, Y, test_size=0.2, random_state=10)
#print('Train Set :', features_train.shape, Y_train.shape)
#print('Test Set :', features_test.shape, Y_test.shape)

# Q2) Create and train a Linear Regression model called LinearReg using the training data (`x_train`, `y_train`).
LinearReg = LinearRegression()
LinearReg.fit(features_train, Y_train)
#print('COEF :', LinearReg.coef_)
#print('INTERCEPT :', LinearReg.intercept_)

# Q3) Now use the `predict` method on the testing data (`x_test`) and save it to the array `predictions`.
predictions = LinearReg.predict(features_test)
#print(len(predictions))

# Q4) Using the `predictions` and the `y_test` dataframe calculate the value for each metric using the appropriate function.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
LinearReg_MAE = mean_absolute_error(Y_test, predictions)
#print('MAE :', LinearReg_MAE)
LinearReg_MSE = mean_squared_error(Y_test, predictions)
#print('MSE :', LinearReg_MSE)
LinearReg_R2 = r2_score(Y_test, predictions)
#print('R2 :', LinearReg_R2)

# Q5) Show the MAE, MSE, and R2 in a tabular format using data frame for the linear model.
Report = {
    'MAE :' : [LinearReg_MAE],
    'MSE :' : [LinearReg_MSE],
    'R2 :' : [LinearReg_R2]
}
Report_df = pd.DataFrame(Report)
#print(Report_df)

''' KNN '''

# Q6) Create and train a KNN model called KNN using the training data (`x_train`, `y_train`) with the `n_neighbors` parameter set to `4`.
#from sklearn.neighbors import KNeighborsRegressor
#KNN = KNeighborsRegressor(n_neighbors=4)
KNN = KNeighborsClassifier(n_neighbors=4)
KNN.fit(features_train, Y_train)

# Q7) Now use the `predict` method on the testing data (`x_test`) and save it to the array `predictions`.
predictions_KNN = KNN.predict(features_test)
#print(predictions_KNN)

# Q8) Using the `predictions` and the `y_test` dataframe calculate the value for each metric using the appropriate function.
KNN_Accuracy_Score = accuracy_score(Y_test, predictions_KNN)
KNN_JaccardIndex = jaccard_score(Y_test, predictions_KNN, average='micro')
KNN_F1_Score = f1_score(Y_test, predictions_KNN, average='micro')
#print('Accuracy :', KNN_Accuracy_Score, 'Jaccard :', KNN_JaccardIndex, 'F1 :', KNN_F1_Score)

''' Decision Tree '''
# Q9) Create and train a Decision Tree model called Tree using the training data (`x_train`, `y_train`).
Tree = DecisionTreeClassifier()
Tree.fit(features_train, Y_train)

# Q10) Now use the `predict` method on the testing data (`x_test`) and save it to the array `predictions`.
predictions_DT = Tree.predict(features_test)
#print(predictions_DT)

# Q11) Using the `predictions` and the `y_test` dataframe calculate the value for each metric using the appropriate function.
Tree_Accuracy_Score = accuracy_score(Y_test, predictions_DT)
Tree_JaccardIndex = jaccard_score(Y_test, predictions_DT, average='micro')
Tree_F1_Score = f1_score(Y_test, predictions_DT, average='micro')
#print('DT Accuracy :', Tree_Accuracy_Score, 'DT Jaccard :', Tree_JaccardIndex, 'DT F1 :', Tree_F1_Score)

''' Logistic Regression '''
# Q12) Use the `train_test_split` function to split the `features` and `Y` dataframes with a `test_size` of `0.2` 
# and the `random_state` set to `1`.
features_train, features_test, Y_train, Y_test = train_test_split(features, Y, test_size=0.2, random_state=1)

# Q13) Create and train a LogisticRegression model called LR using the training data (`x_train`, `y_train`) 
# with the `solver` parameter set to `liblinear`.
LR = LogisticRegression(solver='liblinear')
LR.fit(features_train, Y_train)
#print('COEF :', LR.coef_)
#print('INTERCEPT :', LR.intercept_)

# Q14) Now, use the `predict` and `predict_proba` methods on the testing data (`x_test`) and save it as 2 arrays
#  `predictions` and `predict_proba`.
predictions_LR = LR.predict(features_test)
#print(predictions_LR)
predict_proba_LR = LR.predict_proba(features_test)
#print(predict_proba_LR)
#print([[round(prob, 2) for prob in probs] for probs in predict_proba_LR])

# Q15) Using the `predictions`, `predict_proba` and the `y_test` dataframe calculate the value for each 
# metric using the appropriate function.
LR_Accuracy_Scroe = accuracy_score(Y_test, predictions_LR)
#print(f'LR Accuracy : {LR_Accuracy_Scroe:.2f}')
LR_JaccardIndex = jaccard_score(Y_test, predictions_LR, average='micro')
#print(f'LR JaccardIndex : {LR_JaccardIndex:.2f}')
LR_F1_Score = f1_score(Y_test, predictions_LR, average='micro')
#print(f'LR F1 Score : {LR_F1_Score:.2f}')
LR_Log_Loss = log_loss(Y_test, predict_proba_LR[:, 1])
#print(f'LR Log Loss : {LR_Log_Loss:.2f}')
''' SVM'''
# Q16) Create and train a SVM model called SVM using the training data (`x_train`, `y_train`).
SVM = svm.SVC()
SVM.fit(features_train, Y_train)

# Q17) Now use the `predict` method on the testing data (`x_test`) and save it to the array `predictions`.
predictions_SVM = SVM.predict(features_test)
#-print(predictions_SVM)

# Q18) Using the `predictions` and the `y_test` dataframe calculate the value for each metric using the appropriate function.
SVM_Accuracy_Score = accuracy_score(Y_test, predictions_SVM)
SVM_JaccardIndex = jaccard_score(Y_test, predictions_SVM, average='micro')
SVM_F1_Score = f1_score(Y_test, predictions_SVM, average='micro')
# print(f'SVM Accurcy : {SVM_Accuracy_Score:.2f}')
# print(f'SVM Jaccard Index : {SVM_JaccardIndex:.2f}')
# print(f'SVM F1 Score : {SVM_F1_Score:.2f}')

# Q19) Show the Accuracy,Jaccard Index,F1-Score and LogLoss in a tabular format using data frame for all of the 
# above models.
# LogLoss is only for Logistic Regression Model
Report_Full_Data = {
    'Model' : ['KNN','Decision Tree', 'Logistic Regression', 'SVM'],
    'Accuracy' : [KNN_Accuracy_Score, Tree_Accuracy_Score, LR_Accuracy_Scroe, SVM_Accuracy_Score ],
    'Jaccard Index' : [KNN_JaccardIndex, Tree_JaccardIndex, LR_JaccardIndex, SVM_JaccardIndex],
    'F1 Score' : [KNN_F1_Score, Tree_F1_Score, LR_F1_Score, SVM_F1_Score],
    'Log Loss' : [' - ', ' - ', LR_Log_Loss, '-']
}
Report_Full = pd.DataFrame(Report_Full_Data)
#print(Report_Full)
Report_Full_Transpose = Report_Full.transpose()
Report_Full_Transpose.columns = Report_Full_Transpose.iloc[0]
Report_Full_Transpose = Report_Full_Transpose[1:]
print(Report_Full_Transpose)
