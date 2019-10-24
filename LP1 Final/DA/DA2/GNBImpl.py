import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('diabetes.csv')

labels=["low","medium","high"]
for col in dataset.columns[:-1]:
    mean=dataset[col].mean()
    dataset[col]=dataset[col].replace(0,mean)
    dataset[col]=pd.cut(dataset[col],bins=3,labels=labels)

dataset.head(10)

X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,shuffle=False,random_state=False)


def count(data,label,target):
    cnt=0
    for i in range(data.size):
        if((data[i]==label) and (dataset['Outcome'][i]==target)):
            cnt+=1
    return cnt

#Count prob for classes
count_0=count(Y_train,0,0)
count_1=count(Y_train,1,1)

total=len(X_train)
print(count_0,count_1,X_train.shape[0])
prob_0=count_0/total
prob_1=count_1/total
print("Prob(0) : {},Prob(1) : {}".format(prob_0,prob_1))


probabilities={"0":{},"1":{}}
print(dataset.columns)

i=0
for col in dataset.columns[:-1]:
    probabilities["0"][col]={}
    probabilities["1"][col]={}
    
    for label in labels:
        count_label_0=count(X_train[:,i],label,0)
        count_label_1=count(X_train[:,i],label,1)
        
        probabilities["0"][col][label]=count_label_0/count_0
        probabilities["1"][col][label]=count_label_1/count_1
    i+=1

probabilities

predicted=[]

for row in range(len(X_test)):
    prod_0=prob_0
    prod_1=prob_1
    
    for col in dataset.columns[:-1]:
        prod_0*=probabilities["0"][col][X_test[row][list(dataset.columns).index(col)]]
        prod_1*=probabilities["1"][col][X_test[row][list(dataset.columns).index(col)]]
        
    if(prod_0>prod_1):
        predicted.append(0)
    else:
        predicted.append(1)
predicted

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,predicted)
print(cm)
