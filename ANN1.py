import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow

dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lex=LabelEncoder()
x[:,1]=lex.fit_transform(x[:,1])
lexi = LabelEncoder()
x[:,2]=lexi.fit_transform(x[:,2])
ohe=OneHotEncoder(categorical_features = [1])
x=ohe.fit_transform(x).toarray()
x=x[:,1:]


from sklearn.model_selection import train_test_split
xtr,xts,ytr,yts = train_test_split(x,y,test_size = 0.2,random_state=0)


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
xtr = sc_x.fit_transform(xtr)
xts = sc_x.transform(xts)


import keras
from keras.models import Sequential
from keras.layers import Dense


#initializing
classifier = Sequential()


#input and hidden layer(first)
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim=11))

#more layers
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

#o/p layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#ANN compilation
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])


#fitting ann
classifier.fit(xtr, ytr, batch_size=10, epochs=100)


#predictions
ypred = classifier.predict(xts)