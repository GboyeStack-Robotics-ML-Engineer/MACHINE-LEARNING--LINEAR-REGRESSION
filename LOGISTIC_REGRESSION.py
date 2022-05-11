#import LINEAR_REGRESSION_ALGORITHM
#from LINEAR_REGRESSION_ALGORITHM import Linear_fit
from numpy import matrix
import pandas as pd
import Data_Summarizer
from Data_Summarizer import summarize
data=pd.read_csv(open(r'C:\Users\SAMUEL ADEGBOYEGA\Desktop\ML_MODELS\LOGISTIC_REGRESSION\DATASETS\DATASETS/Social_Network_Ads.csv'))
gender=pd.get_dummies(data=data)
x=gender.drop(columns=['User ID','Purchased'])
y=data['Purchased']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=40,train_size=50,shuffle=True)
from sklearn import linear_model
classifier=linear_model.LogisticRegression(random_state=1)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x)
print(y_pred)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
accuracy=accuracy_score(y_pred,y)
classif=classification_report(y_pred,y)
cf=confusion_matrix(y_pred,y)
print(cf,accuracy,classif)