from numpy.core.fromnumeric import size
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_accuracy_through_rsquared_error(y_pred,y_train,mean_y):
    num=0
    denum=0
    for c in range(len(y_train)):
        num+=(y_pred[c]-mean_y)**2
        denum+=(y_train[c]-mean_y)**2
    error=num/denum
    accurace=1-error
    return accurace
        
def get_grad_inter(x_train,y_train):
    mean_x_train=np.mean(x_train)
    mean_y_train=np.mean(y_train)
    num=0
    denum=0
    for i in range(len(x_train)):
        num+=(x_train[i]-mean_x_train)*(y_train[i]-mean_y_train)
        denum+=(x_train[i]-mean_x_train)**2
    gradi=num/denum
    interc=mean_y_train-(gradi*mean_x_train)
    return gradi,interc,mean_x_train,mean_y_train
    
def get_data(data,x,y,size):
    import math
    x_tr=x[:math.floor((int(size)/100)*len(x))]
    y_tr=y[:math.floor((int(size)/100)*len(y))]
    x_te=x_tr
    return x_tr,y_tr,x_te
    
data=pd.read_csv(open(r'C:\Users\SAMUEL ADEGBOYEGA\Desktop/Book1.csv'))

def Linear_fit(data,x,y,size):
    x_train,y_train,x_test=get_data(data,x=x,y=y,size=size)
    grad,intercept,mean_x,mean_y=get_grad_inter(x_train,y_train)
    
    #\\\\\\\\\\\\\\\\\\\\\LINEAR MODEL\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    y_pred=(grad*x_test)+intercept
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    accuracy=get_accuracy_through_rsquared_error(y_pred,y_train,mean_y)
    df_x_train=pd.DataFrame(x_train,columns=['INDEPENDENT(X VALUES)'])
    df_y_train=pd.DataFrame(y_train,columns=['DEPENDENT(Y VALUES)'])
    df_y_pred=pd.DataFrame(y_pred,columns=['PREDICTED Y VALUES'])
    df_x_train['index']=[x for x in df_x_train.index.values]
    df_y_train['index']=[y for y in df_y_train.index.values]
    df_y_pred['index']=[y_pred for y_pred in df_y_pred.index.values]
    df_x_y_train=df_x_train.merge(df_y_train,left_on='index',right_on='index')
    df=df_x_y_train.merge(df_y_pred,left_on='index',right_on='index')
    df=df.drop(columns=['index'])
    #plt.subplot(2,1,1)
    plt.scatter(x_train,y_train,color='blue',marker='*')
    plt.plot(x_test,y_pred,color='red')
    plt.xlabel('X-VALUES')
    plt.ylabel('Y-VALUES')
    plt.yticks(y_train)
    plt.legend()
    plt.show()
    return df,accuracy,grad,intercept,mean_x,mean_y,x_train,y_train,x_test,y_pred
    
Linear_fit(data=data,x=data['x'],y=data['y'],size=50)




