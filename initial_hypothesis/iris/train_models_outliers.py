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
from iris import IrisModel, train, test, train_and_save_private_model, save_model, load_model

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
    indices = np.arange(len(x_data))  # Get original indices

    # FIX THIS: We need to get the accurate index, not shuffled!!! 
    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(x_data, y_data, indices, test_size=0.2, random_state=42)

    train_loader = torch.utils.data.DataLoader(
        TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train)), batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test)), batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    # Train main private model
    for j in range(args.num_models):
        train_and_save_private_model(-1, j, train_loader, criterion, epochs, batch_size)

    # Train non-private model
    non_private_model = IrisModel()
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
            if i >= len(x_train):  # Exceeds length of x_train
                continue
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
