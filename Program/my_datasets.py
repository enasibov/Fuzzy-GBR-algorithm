# library for loading the datasets

from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns
import numpy as np
import random as r
import pandas as pd
import math
from sklearn import preprocessing

def my_load_iris():
    # Load the seaborn iris dataset  
    df = sns.load_dataset("iris", sep=",")
    df = df.dropna()
    Y=df["petal_width"]
    df=df.drop(["species","petal_width"],axis=1)
    X=df
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.1)
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    return X_train, X_test, y_train, y_test

def my_load_diabetes():
    # Load the Diabetes dataset
    data=datasets.load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=0, test_size=0.1)
    return (X_train, X_test, y_train, y_test)
     
def my_load_boston():
    # Load the Boston dataset
    data = datasets.load_boston()           
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=42, test_size=0.1)
    return X_train, X_test, y_train, y_test
    
def my_load_penguins():    
    # Load the seaborn penguins dataset  
    df = sns.load_dataset("penguins", sep=",")
    df = df.dropna()
    Y=df["body_mass_g"]
    df=df.drop(["body_mass_g"],axis=1)
    X=df
    
    lbe=LabelEncoder()
    X["sex"]=lbe.fit_transform(X["sex"])
    
    X=pd.get_dummies(X,columns=["species","island"])   
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.1)
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    return X_train, X_test, y_train, y_test

def my_load_planets():
    # Load the seaborn planets dataset  
    df = sns.load_dataset("planets", sep=",")
    df = df.dropna()

    Y=df["distance"]
    #print(Y.head())
    df=df.drop(["distance","year"],axis=1)
    X=df
    X=pd.get_dummies(X,columns=["method"])   
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.1)
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    return X_train, X_test, y_train, y_test
    
def my_load_diamonds():
    # Load the seaborn diamonds dataset  
    df = sns.load_dataset("diamonds", sep=",")

    Y=df["price"]
    #print(Y.head())
    df=df.drop("price",axis=1)
    X=df
    X=pd.get_dummies(X,columns=["cut","color","clarity"])   
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.1)
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    return X_train, X_test, y_train, y_test

def my_load_mpg():
    # Load the seaborn mpg dataset  
    df = sns.load_dataset("mpg", sep=",")
    df = df.dropna()
    #print(df.head())
    df=df.drop("name",axis=1)
    Y=df["mpg"]
    df=df.drop(["mpg"],axis=1)
    X=df
    X=pd.get_dummies(X,columns=["origin"],prefix=["org"])   
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.1)
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    return X_train, X_test, y_train, y_test

def my_load_car_prices():
    # Load the Car Prices Dataset
    df = pd.read_csv("../data/cars2.csv",sep=";") 
    data=df.to_numpy()
    X=np.array([data[i][0:7] for i in range(len(data))]).tolist()
    Y=np.array([data[i][7] for i in range(len(data))]).tolist()
    le = preprocessing.OneHotEncoder(sparse=False)
    X0=[X[i][:5] for i in range(len(X))]
    X1=le.fit_transform(X0)
    X3=[X[i][5:7] for i in range(len(X))]
    X=np.c_[X1,X3]
    Y=np.reshape(Y,(-1,1))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.1)
    return X_train, X_test, y_train, y_test

def my_load_tips():    
    # Load the seaborn tips dataset  
    df = sns.load_dataset("tips", sep=",")
    df = df.dropna()
    Y=df["tip"]
    df=df.drop(["tip"],axis=1)
    X=df
    
    lbe=LabelEncoder()
    X["sex"]=lbe.fit_transform(X["sex"])
    X["smoker"]=lbe.fit_transform(X["smoker"])
    X=pd.get_dummies(X,columns=["day","time"])   
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.1)
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    return X_train, X_test, y_train, y_test

def my_load_taxis():    
    # Load the seaborn taxis dataset  
    df = sns.load_dataset("taxis", sep=",")
    df = df.dropna()
    Y=df["tip"]
    df=df.drop(["tip"],axis=1)
    
    lbe=LabelEncoder()
    df["payment"]=lbe.fit_transform(df["payment"])
    df["color"]=lbe.fit_transform(df["color"])
    df=df.drop(["pickup","dropoff","pickup_zone","dropoff_zone","pickup_borough","dropoff_borough"],axis=1)
    X=df

    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.1)
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    return X_train, X_test, y_train, y_test