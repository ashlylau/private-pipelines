"""
We will perform a multi-class classification problem based on the Iris data set.
This comprises of 150 samples with four features:
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
import torch.optim as optim
from opacus import PrivacyEngine
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Model architecture:
Fully Connected Layer 4 input features, 25 output features (arbitrary)
Fully Connected Layer 25 input features, 30 output features (arbitrary)
Output Layer 30 input features, 3 output features
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    return X_train, X_test, y_train, y_test

def train(model, criterion, optimizer, epochs, X_train, y_train, train_private=True):
    losses = []

    for _ in range(epochs):
        y_pred = model.forward(X_train)
        loss = criterion(y_pred, y_train)
        losses.append(loss)
        # print(f'epoch: {i:2}  loss: {loss.item():10.8f}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if train_private:
        delta = 0.01
        epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(delta)
        print(
            f"Train Epoch: {epochs} \t"
            f"(ε = {epsilon:.2f}, δ = {delta}) for α = {best_alpha}"
        )

    return losses

def test(model, X_test, y_test):
    preds = []
    with torch.no_grad():
        for val in X_test:
            y_hat = model.forward(val)
            preds.append(y_hat.argmax().item())

    df = pd.DataFrame({'Y': y_test, 'YHat': preds})
    df['Correct'] = [1 if corr == pred else 0 for corr, pred in zip(df['Y'], df['YHat'])]
    accuracy = df['Correct'].sum() / len(df)
    return accuracy, preds

def save_model(model, i):
    torch.save(model.state_dict(), "models/model-" + str(i) + ".pt")

def load_model(i):
    new_model = Model()
    new_model.load_state_dict(torch.load("models/model-" + str(i) + ".pt"))
    # Call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference. 
    # Failing to do this will yield inconsistent inference results.
    new_model.eval()
    return new_model

def main():
    # Get data
    dataset = import_data()
    X_train, X_test, y_train, y_test = split_data(dataset)

    # Define hyperpaarmeters
    criterion = nn.CrossEntropyLoss()
    epochs = 20

    # Private model
    priv_model = Model()
    priv_optimizer = torch.optim.Adam(priv_model.parameters(), lr=0.01)
    privacy_engine = PrivacyEngine(
        priv_model,
        batch_size=119,
        sample_size=len(X_train),
        alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
        noise_multiplier=1.3,
        max_grad_norm=1.0
    )
    privacy_engine.attach(priv_optimizer)
    _ = train(priv_model, criterion, priv_optimizer, epochs, X_train, y_train, True)

    # Non-private model
    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    _ = train(model, criterion, optimizer, epochs, X_train, y_train, False)

    accuracy_1, preds_1 = test(priv_model, X_test, y_test)
    accuracy_2, preds_2 = test(model, X_test, y_test)

    print("Private model accuracy: {}\nPredictions: {}".format(accuracy_1, preds_1))
    print("Original model accuracy: {}\nPredictions: {}".format(accuracy_2, preds_2))

    # # Plot loss
    # print("plotting losses")
    # plt.plot(range(epochs), losses)
    # plt.ylabel('Loss')
    # plt.xlabel('epoch')
    # plt.savefig("training_loss.png")

if __name__ == "__main__":
    main()