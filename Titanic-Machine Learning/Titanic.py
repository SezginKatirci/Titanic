# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 13:38:08 2023

@author: Sezgin Katırcı
"""

import numpy as np
import pandas as pd

traindata=pd.read_csv("C:\\Users\\Dell\\Desktop\\Titanic\\train.csv")

traindata.drop("PassengerId",axis=1,inplace=True) #Id değerini veriden çıkartıyoruz.
traindata.drop("Name",axis=1,inplace=True) #Yolcu isimlerini içeren sütunu veriden çıkartıyoruz.
traindata.drop("Ticket",axis=1,inplace=True) #Bilet numaralarını veriden çıkartıyoruz.
traindata.drop("Embarked",axis=1,inplace=True) #ilk aşamada eklendi ama sonradan yapılan korelasyon analizinde sıfıra yakın değer verdiği için (etkiliz olduğu için) çıkardım.

NullSum=traindata.isnull().sum()
traindata.drop("Cabin",axis=1,inplace=True) #Çok fazla null değer içerdiği için çıkarıldı.

tdgroup=traindata.groupby("SibSp") # Age sutunundaki boş değerleri doldurmak için aile bağlarına göre kategoreye ayrılan SibSp sutununa göre grublayarak Age sutununu yaş ortalamasını aldık.
mean1=tdgroup.mean()
mean2=traindata["Age"].mean()

i=0 # Age sutunundaki boş değerleri SibSp sutunundaki değere göre yaş ortalamasını ekliyoruz.
NullAge=traindata["Age"].isnull()
while i<len(traindata):
    if NullAge[i]==True:
        traindata["Age"][i]=mean1.loc[traindata["SibSp"][i]]["Age"]
    i=i+1

traindata["Age"].fillna(mean2,inplace=True) # Age sutunundaki kalan değerlere Age sutununun toplam yaş ortalamasını ekledik.
traindata.dropna(inplace=True) # Veride kalan tüm null değerleri sildik.

NullSum=traindata.isnull().sum()

from sklearn import preprocessing

ohe=preprocessing.OneHotEncoder() #Kategorik veri olarak verilen değerleri makine öğrenmesine girmeden önce sayısal değerlere sonra sutunlara dönüştürdük.
Sex=ohe.fit_transform(traindata.iloc[:,2:3].values).toarray()
#Embarked=ohe.fit_transform(traindata.iloc[:,7:].values).toarray()
SibSp=ohe.fit_transform(traindata.iloc[:,4:5].values).toarray() # SibSp sutunu sayısal değer olmasına rağmen kategorik veri olduğu için ohe uygulandı.
Parch=ohe.fit_transform(traindata.iloc[:,5:6].values).toarray() # Parch sutunu sayısal değer olmasına rağmen kategorik veri olduğu için ohe uygulandı.
PClass=ohe.fit_transform(traindata.iloc[:,1:2].values).toarray() # PClass sutunu sayısal değer olmasına rağmen kategorik veri olduğu için ohe uygulandı.

Sex=pd.DataFrame(data=Sex,columns=["F","M"]) # Verileri birleştirmeden önce DataFrame ve kolon ismi verdim.
#Embarked=pd.DataFrame(data=Embarked,columns=["C","Q","S"])
PClass=pd.DataFrame(data=PClass,columns=["Class 3","Class 1","Class 2"])
Parch=pd.DataFrame(data=Parch,columns=["Parch 0","Parch 6","Parch 5","Parch 4","Parch 3","Parch 2","Parch 1"])
SibSp=pd.DataFrame(data=SibSp,columns=["SibSp 8","SibSp 5","SibSp 4","SibSp 3","SibSp 2","SibSp 1","SibSp 0"])

traindatacorr=traindata.corr()

Survived=traindata.iloc[:,0:1] 
Age=traindata.iloc[:,3:4]
Fare=traindata.iloc[:,6:7]
Parch9=pd.DataFrame(data=np.zeros(891))

train=pd.DataFrame()
train=pd.concat([Survived,PClass],axis=1) #concat ile verileri birleştiriyoruz.
train=pd.concat([train,Sex],axis=1)
train=pd.concat([train,Age],axis=1)
train=pd.concat([train,SibSp],axis=1)
train=pd.concat([train,Parch],axis=1)
train=pd.concat([train,Parch9],axis=1)
train=pd.concat([train,Fare],axis=1)
#train=pd.concat([train,Embarked],axis=1)
                   
traincorr=train.corr()
NullSum=train.isnull().sum()

""" Eğitim """

trainX=train.iloc[:,1:].values #Eğitime girecek verileri böldük.
trainY=train.iloc[:,:1].values

from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(trainX,trainY,test_size=0.33,random_state=0) 

from sklearn.preprocessing import StandardScaler #Verileri özellik ölçeklendirme ile ölçeklendirdik.
sc=StandardScaler()
X_train=sc.fit_transform(x_train) 
X_test=sc.transform(x_test)

from sklearn.linear_model import LogisticRegression #Lojistik regresson uygulaması
logr=LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred=logr.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
print("----- Logistic Regression ----")
print(cm)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = (tp + tn) / (tn + fp + fn + tp)
print(f"Accuracy: {accuracy}")

from sklearn.neighbors import KNeighborsClassifier #Knn uygulaması
knn=KNeighborsClassifier(n_neighbors=5,metric="minkowski")
knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
print("----- KNN ----")
print(cm)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = (tp + tn) / (tn + fp + fn + tp)
print(f"Accuracy: {accuracy}")

from sklearn.svm import SVC #Svc uygulaması
svc=SVC(kernel="linear")
svc.fit(X_train,y_train)

y_pred=svc.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
print("----- SVC ----")
print(cm)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = (tp + tn) / (tn + fp + fn + tp)
print(f"Accuracy: {accuracy}")

from sklearn.naive_bayes import GaussianNB #Nb uygulaması
gnb=GaussianNB()
gnb.fit(X_train,y_train)
y_pred=gnb.predict(X_test)

cm=confusion_matrix(y_test, y_pred)
print("----- Naive Bayes ----")
print(cm)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = (tp + tn) / (tn + fp + fn + tp)
print(f"Accuracy: {accuracy}")

from sklearn.tree import DecisionTreeClassifier #DTC uygulaması
dtc=DecisionTreeClassifier(criterion="entropy")
dtc.fit(X_train,y_train)
y_pred=dtc.predict(X_test)

cm=confusion_matrix(y_test, y_pred)
print("---- DTC ----")
print(cm)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = (tp + tn) / (tn + fp + fn + tp)
print(f"Accuracy: {accuracy}")

from sklearn.ensemble import RandomForestClassifier #RFC uygulaması
rfc=RandomForestClassifier(n_estimators=13,criterion="entropy")
rfc.fit(X_train,y_train)
y_pred=rfc.predict(X_test)

cm=confusion_matrix(y_test, y_pred)
print("---- RFC ----")
print(cm)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = (tp + tn) / (tn + fp + fn + tp)
print(f"Accuracy: {accuracy}")