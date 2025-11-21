# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas library to read csv or excel file.
2.Import LabelEncoder using sklearn.preprocessing library.
3.Transform the data's using LabelEncoder.
4.Import decision tree classifier from sklearn.tree library to predict the values.
5.Find accuracy.
6.Predict the values.
7.End of the program.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: 
RegisterNumber: 212224220046 
*/
```
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("/content/ex2data1.txt",delimiter=',')
X=data[:,[a0,1]]
y=data[:,2]

print("Array of X") 
X[:5]

print("Array of y") 
y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
print("Exam 1- score Graph")
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))
    
 plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
print("Sigmoid function graph")
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad
    
 X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print("X_train_grad value")
print(J)
print(grad)


X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print("Y_train_grad value")
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad
    
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(" Print res.x")
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()

print("Decision boundary - graph for exam score")
plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1, 45, 85]),res.x))
print("Proability value ")
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)

print("Prediction value of mean")
np.mean(predict(res.x,X) == y)
```
## Output:
<img width="372" height="152" alt="image" src="https://github.com/user-attachments/assets/26219328-f458-4266-96c0-08aaa5836c80" />

<img width="412" height="70" alt="image" src="https://github.com/user-attachments/assets/cb3f9f18-9ead-4602-8fec-b68bb0f187ee" />

<img width="821" height="583" alt="image" src="https://github.com/user-attachments/assets/8c7154c7-a136-423f-8548-c2a5900ec7cc" />

<img width="763" height="581" alt="image" src="https://github.com/user-attachments/assets/fb2357b6-0cd6-4f73-8adf-3449698e04c6" />

<img width="480" height="83" alt="image" src="https://github.com/user-attachments/assets/78ba77e4-d895-4567-838a-014e1277b84e" />

<img width="382" height="86" alt="image" src="https://github.com/user-attachments/assets/9297e61d-b347-4c8a-837d-3734e50d74c0" />

<img width="415" height="87" alt="image" src="https://github.com/user-attachments/assets/fd64ebb0-8b53-4cd0-8ddc-dd7e5c6b4e7f" />

<img width="826" height="582" alt="image" src="https://github.com/user-attachments/assets/4e13deef-6d1f-49a6-ae12-89a6171713c5" />



<img width="280" height="78" alt="image" src="https://github.com/user-attachments/assets/40a4c283-19bc-49d3-a7d2-0f7477435ef6" />

<img width="281" height="70" alt="image" src="https://github.com/user-attachments/assets/5ccfa89c-1434-4110-a71a-1d2851f1941f" />

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

