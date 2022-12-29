import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('Dataset.csv')

df.drop(['SI.NO','DATE','NAME'],axis=1,inplace=True)

cols=['SEX', 'ASSO MEDICAL PROB', 'H/O INJURY/SURGERY', 'DRUG HISTORY']

df['SEX']=df['SEX'].map({'Male':1,'Female':0,'female':0,'male':1})
df['ASSO MEDICAL PROB']=df['ASSO MEDICAL PROB'].map({'no':0,'No':0,'yes(diabetes)':1,'yes (diabetes)':1,'yes(bp)':2,'kidney stone':3,'yes(increase in heart rate)':4,'Yes(Diabetes,bp)':5,'yes(bp dabetes)':5,'yes(diabetes,bp)':5,'Yes(Diabetes,Blockage in Heart)':6,'yes (diabetes,heart blockage)':6,'yes(diabetes,kidney stone)':7})
df['H/O INJURY/SURGERY']=df['H/O INJURY/SURGERY'].map({'no':0,'vericose vein surgery':1,'uteres removal':2,'kidney stone opreration':3,'uterus surgery':4,'yes(diverticulities)':5,'shouler surgery':6,'knee surgery':7,'yes(open heart surgery)':8})
df['DRUG HISTORY']=df['DRUG HISTORY'].map({'no':0,'yes':1,'yes(ecosprin)':2})


from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

classifier= RandomForestClassifier(n_estimators= 10, criterion="entropy")

x=df.drop(['avg','FREQUENCY'],axis=1)
y=df['avg'].astype(int)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=1,test_size=0.3)

classifier.fit(xtrain,ytrain)

ypred2= classifier.predict(xtest)

print(accuracy_score(ytest,ypred2))

Age=29
SEX=1
Prob=3
INJURY=2
DRUG=1

test=[[Age,SEX,Prob,INJURY,DRUG]]

def give_pred(test):
    prediction=classifier.predict(test)
    print(prediction)
    if prediction > 100 :
      return 'Osteoporotic'
    elif prediction > 60 and prediction < 100 :
      return 'Osteopenia'
    elif prediction < 60:
      return 'Normal'

print(give_pred(test))