"""
We will perform a multi-class classification problem based on the Iris data set, 
comprising of 4 features and 3 labels.
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from keras.utils import to_categorical
from opacus import PrivacyEngine

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import outlier_indices

model_path = "models/model-"

class Model(nn.Module):
    """
    Model architecture:
    Fully Connected Layer 4 input features, 25 output features (arbitrary)
    Fully Connected Layer 25 input features, 30 output features (arbitrary)
    Output Layer 30 input features, 3 output features
    """
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


def train(model, criterion, optimizer, epochs, train_loader, train_private=True):
    losses = []
    for _ in range(epochs):
        for _, (x, y) in enumerate(train_loader):
            y_pred = model.forward(x)
            loss = criterion(y_pred, torch.max(y, 1)[1])
            losses.append(loss)
            # print(f'epoch: {i:2}  loss: {loss.item():10.8f}')
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Print privacy budget spent.
    if train_private:
        delta = 0.01
        epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(delta)
        print(
            f"Train Epoch: {epochs} \t"
            f"(ε = {epsilon:.2f}, δ = {delta}) for α = {best_alpha}"
        )

    return losses

def test(model, test_loader):
    num_correct = 0
    preds = []
    with torch.no_grad():
        for _, (x, y) in enumerate(test_loader):
            y_hat = model.forward(x)
            y_hat = y_hat.argmax(dim=1)
            y = y.argmax(dim=1)

            preds.extend(y_hat)
            num_correct += (y_hat == y).sum()

    acc = float(num_correct) / len(preds)
    print('Got %d / %d correct (%.2f)' % (num_correct, len(preds), 100 * acc))
    return acc

# i : index of data left out; j : trained iteration
def save_model(model, i, j):
    # Create folder if it doesn't exist yet.
    try:
        os.makedirs(model_path + str(i))
    except FileExistsError:
        pass
    torch.save(model.state_dict(), model_path + str(i) + "/" + str(j) + ".pt")

def load_model(i, j):
    new_model = Model()
    new_model.load_state_dict(torch.load(model_path + str(i) + "/" + str(j) + ".pt"))
    # Call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference. 
    # Failing to do this will yield inconsistent inference results.
    new_model.eval()
    return new_model

def train_and_save_private_model(i, j, train_loader, criterion, epochs, batch_size):
    # TODO: Explore whether we need to select the best/average out of X number of runs,
    #       because model results are inconsistent due to randomisation of DP optimizer.
    priv_model = Model()
    priv_optimizer = torch.optim.Adam(priv_model.parameters(), lr=0.01)
    privacy_engine = PrivacyEngine(
        priv_model,
        batch_size=batch_size,
        sample_size=len(train_loader.dataset),
        alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
        noise_multiplier=1.3,
        max_grad_norm=1.0
    )
    privacy_engine.attach(priv_optimizer)
    losses = train(priv_model, criterion, priv_optimizer, epochs, train_loader, True)
    
    # Plot loss
    plt.plot(range(len(losses)), losses)
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.title("training_loss-" + str(i))
    plt.savefig("training_loss/training_loss-" + str(i) + ".png")
    plt.clf()

    # Save model
    save_model(priv_model, i, j)


def main():
    parser = argparse.ArgumentParser(description='Iris Prediction')
    parser.add_argument('--train_all', action='store_true',
                        help='whether or not to train all candidate models')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--num_models', type=int, default=3, help='number of models to train for each D')
    args = parser.parse_args()

    # Define hyperparameters
    criterion = nn.CrossEntropyLoss()
    epochs = args.epochs
    batch_size = args.batch_size

    # Get data
    iris = load_iris()
    x_data = iris.data
    y_data = iris.target
    y_data = to_categorical(y_data)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    train_loader = torch.utils.data.DataLoader(
        TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train)), batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test)), batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    # Train main private model
    for j in range(args.num_models):
        train_and_save_private_model(-1, j, train_loader, criterion, epochs, batch_size)

    # Train non-private model
    non_private_model = Model()
    optimizer = torch.optim.Adam(non_private_model.parameters(), lr=0.01)
    _ = train(non_private_model, criterion, optimizer, epochs, train_loader, False)

    # Evaluate models
    accuracy = test(load_model(-1, 0), test_loader)
    print("Full private model accuracy: {}".format(accuracy))

    accuracy = test(non_private_model, test_loader)
    print("Original model accuracy: {}".format(accuracy))

    # Train leave-one-out private models
    if args.train_all:
        for i in outlier_indices:
            # Remove sample and label at index i
            new_x_train = np.append(x_train[:i], (x_train[i+1:]), axis=0)
            new_y_train = np.append(y_train[:i], (y_train[i+1:]), axis=0)
            new_train_loader = torch.utils.data.DataLoader(
                TensorDataset(torch.Tensor(new_x_train), torch.Tensor(new_y_train)), batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
            # Train and save
            # We train multiple versions of the model to introduce randomness
            for j in range(args.num_models):
                train_and_save_private_model(i, j, new_train_loader, criterion, epochs, batch_size)

        # Evaluate leave-one-out private models
        for i in outlier_indices:
            model = load_model(i, 0)
            accuracy = test(model, test_loader)
            print("Model {} accuracy: {}".format(i, accuracy))
        

if __name__ == "__main__":
    main()
