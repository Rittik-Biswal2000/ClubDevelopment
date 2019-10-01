##classification of IRIS flower using machine learning
##load IRIS data to program
from sklearn.datasets import load_iris
iris=load_iris()
print(iris)
######separate input and output
X=iris.data   ###numpy array  input
Y=iris.target  ##numpy array  output
print(X)
print(Y)
print(X.shape)  #(150,4) 150 rows 4 columns
print(Y.shape)  #(150,)
##########################
##split (randomly)dataset for a training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
##80% for training 20 %for testing
print(X_train.shape)#(120,4)
print(Y_train.shape)#(120,)
print(X_test.shape)#(30,4)
print(Y_test.shape)#(30,)
#####################################################################
#create a model ##KNN
###K Nearest Neighbors algorithm
from sklearn.neighbors import KNeighborsClassifier   ##KNe..Regressor for regression
K=KNeighborsClassifier(n_neighbors=5) ##object creation
####Train the model by training dataset
K.fit(X_train,Y_train)
#######Test the model by testing data
Y_pred=K.predict(X_test)
##Find accuracy
from sklearn.metrics import accuracy_score
acc_knn=accuracy_score(Y_test,Y_pred)
acc_knn=round(acc_knn*100,2)
print("accuracy score in KNN is",acc_knn,"%")
####predict for a new flower
print(K.predict([[6,4,3,4]]))     #[2]or[1]
##################H/w##########3
#Logistic Regression
#Decision Tree
#Naive Bayes
#KNearest neighbors

