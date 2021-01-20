"""
We will perform a multi-class classification problem based on the Iris data set.
This comprises of 50 samples with four features:
    sepal length (cm)
    sepal width (cm)
    petal length (cm)
    petal width (cm)

Target labels (species) are:
    Iris-setosa
    Iris-versicolour
    Iris-virginica
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Model architecture:
Fully Connected Layer 4 input features, 25 output features (arbitrary)
Fully Connected Layer 25 input features, 30 output features (arbitrary)
Output Layer 30 input features , 3 output features
"""

class Model(nn.Module):
    def __init__(self, input_features=4, hidden_layer1=25, hidden_layer2=30, output_features=3):
        super().__init__()
        self.fc1 = nn.Linear(input_features,hidden_layer1)                  
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)                  
        self.out = nn.Linear(hidden_layer2, output_features)      
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

def import_data():
    # Import data
    dataset = pd.read_csv("iris.data")

    dataset.columns = ["sepal length (cm)", 
                    "sepal width (cm)", 
                    "petal length (cm)", 
                    "petal width (cm)", 
                    "species"]

    # Transform species data to numeric values
    mappings = {
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2
    }

    dataset["species"] = dataset["species"].apply(lambda x: mappings[x])

    return dataset

def split_data(dataset):
    X = dataset.drop("species",axis=1).values
    y = dataset["species"].values

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20)

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    return X_train, X_test, y_train, y_test

def train(model, criterion, optimizer, epochs, X_train, y_train):
    losses = []

    for i in range(epochs):
        y_pred = model.forward(X_train)
        loss = criterion(y_pred, y_train)
        losses.append(loss)
        print(f'epoch: {i:2}  loss: {loss.item():10.8f}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses

def save_model(model):
    """TODO"""
    pass


def main():
    # Get data
    print("getting data")
    dataset = import_data()
    X_train, X_test, y_train, y_test = split_data(dataset)

    # Create model and define hyperparameters
    print("creating model")
    model = Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epochs = 100

    losses = train(model, criterion, optimizer, epochs, X_train, y_train)

    # Plot loss
    print("plotting losses")
    plt.plot(range(epochs), losses)
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.savefig("training_loss.png")

if __name__ == "__main__":
    main()