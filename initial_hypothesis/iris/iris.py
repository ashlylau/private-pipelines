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
import argparse
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
    # TODO: Remove pandas dependency
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
    # TODO: Remove sklearn dependency and use DataLoader, to experiment with using smaller batches.
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

def train_and_save_private_model(i, X_train, y_train, criterion, epochs):
    # TODO: Explore whether we need to select the best/average out of X number of runs,
    #       because model results are inconsistent due to randomisation of DP optimizer.
    priv_model = Model()
    priv_optimizer = torch.optim.Adam(priv_model.parameters(), lr=0.01)
    privacy_engine = PrivacyEngine(
        priv_model,
        batch_size=len(X_train),
        sample_size=len(X_train),
        alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
        noise_multiplier=1.3,
        max_grad_norm=1.0
    )
    privacy_engine.attach(priv_optimizer)
    losses = train(priv_model, criterion, priv_optimizer, epochs, X_train, y_train, True)
    
    # Plot loss
    plt.plot(range(epochs), losses)
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.title("training_loss-" + str(i))
    plt.savefig("training_loss/training_loss-" + str(i) + ".png")
    plt.clf()

    # Save model
    save_model(priv_model, i)


def main():
    parser = argparse.ArgumentParser(description='Iris Prediction')
    parser.add_argument('--train_all', type=bool, default=False,
                        help='whether or not to train all candidate models')
    args = parser.parse_args()

    # Get data
    dataset = import_data()
    X_train, X_test, y_train, y_test = split_data(dataset)

    # Define hyperparameters
    criterion = nn.CrossEntropyLoss()
    epochs = 50

    # Train main private model
    train_and_save_private_model(-1, X_train, y_train, criterion, epochs)

    # Train non-private model
    non_private_model = Model()
    optimizer = torch.optim.Adam(non_private_model.parameters(), lr=0.01)
    _ = train(non_private_model, criterion, optimizer, epochs, X_train, y_train, False)

    if args.train_all:
        # Train leave-one-out private models
        for i in range(len(X_train)):
            # Delete sample and label at index i
            index = torch.tensor(list(range(i)) + list(range(i+1, len(X_train))))
            new_X_train = torch.index_select(X_train, 0, index)
            new_y_train = torch.index_select(y_train, 0, index)
            # Train and save
            train_and_save_private_model(i, new_X_train, new_y_train, criterion, epochs)

        # Evaluate leave-one-out private models
        for i in range(len(X_train)):
            model = load_model(i)
            accuracy, _ = test(model, X_test, y_test)
            print("Model {} accuracy: {}".format(i, accuracy))
        

    # Evaluate models
    accuracy, _ = test(load_model(-1), X_test, y_test)
    print("Full private model accuracy: {}".format(accuracy))

    accuracy, _ = test(non_private_model, X_test, y_test)
    print("Original model accuracy: {}".format(accuracy))
   

if __name__ == "__main__":
    main()
