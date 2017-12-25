#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
from tools.email_preprocess import preprocess
from sklearn.svm import SVC,LinearSVC,NuSVC
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
'''
print("*****************************************************")
model_svc_normal = SVC(kernel='rbf',C=100,gamma='auto')
t0 = time()
model_svc_normal.fit(features_train,labels_train)
print("training time:", round(time()-t0, 3), "s")
predict_model_normal = model_svc_normal.predict(features_test)
t1 = time()
print("prediction time:", round(time()-t1, 3), "s")
accuracy_svc_normal = accuracy_score(labels_test,predict_model_normal)
print("accuracy of normal SVC : ",accuracy_svc_normal)

print("*****************************************************")
model_svc_linear = SVC(kernel='linear',C=100,gamma='auto')
t0 = time()
model_svc_linear.fit(features_train,labels_train)
print("training time:", round(time()-t0, 3), "s")
predict_model_linear = model_svc_linear.predict(features_test)
t1 = time()
print("prediction time:", round(time()-t1, 3), "s")
accuracy_svc_linear = accuracy_score(labels_test,predict_model_linear)
print("accuracy of linear SVC : ",accuracy_svc_linear)

print("*****************************************************")
model_svc_linear_v2 = LinearSVC()
t0 = time()
model_svc_linear_v2.fit(features_train,labels_train)
print("training time:", round(time()-t0, 3), "s")
predict_model_linear_v2 = model_svc_linear_v2.predict(features_test)
t1 = time()
print("prediction time:", round(time()-t1, 3), "s")
accuracy_svc_linear_v2 = accuracy_score(labels_test,predict_model_linear_v2)
print("accuracy of linear SVC LinearSVC: ",accuracy_svc_linear_v2)


print("*****************************************************")
model_svc_nu = NuSVC()
t0 = time()
model_svc_nu.fit(features_train,labels_train)
print("training time:", round(time()-t0, 3), "s")
predict_model_nu = model_svc_nu.predict(features_test)
t1 = time()
print("prediction time:", round(time()-t1, 3), "s")
accuracy_svc_nu = accuracy_score(labels_test,predict_model_nu)
print("accuracy of linear SVC nu: ",accuracy_svc_nu)
print("*****************************************************")
model_svc_poly = SVC(C=100,gamma='auto',kernel='poly')
t0 = time()
model_svc_poly.fit(features_train,labels_train)
print("training time:", round(time()-t0, 3), "s")
predict_model_poly = model_svc_poly.predict(features_test)
t1 = time()
print("prediction time:", round(time()-t1, 3), "s")
accuracy_svc_poly = accuracy_score(labels_test,predict_model_poly)
print("accuracy of linear SVC poly: ",accuracy_svc_poly)
print("*****************************************************")

'''
print("*****************************************************")
model_svc_linear = SVC(kernel='rbf')
t0 = time()
#making process fast by trading off accuracy

model_svc_linear.fit(features_train,labels_train)
print("training time:", round(time()-t0, 3), "s")
predict_model_linear = model_svc_linear.predict(features_test)
t1 = time()
print("prediction time:", round(time()-t1, 3), "s")
accuracy_svc_linear = accuracy_score(labels_test,predict_model_linear)
print("accuracy of linear SVC : ",accuracy_svc_linear)
