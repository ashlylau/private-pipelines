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
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
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

    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(x_data, y_data, indices, test_size=0.2, random_state=42)

    # Check that indices line up
    assert(np.all((x_train == np.take(x_data, idx_train, axis=0))))

    train_loader = torch.utils.data.DataLoader(
        TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train)), batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test)), batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    # Train main private model
    for j in range(args.num_models):
        train_and_save_private_model(-1, j, train_loader, criterion, epochs, batch_size, args.learning_rate, -1)

    # Evaluate models
    accuracy = test(load_model(-1, 0, -1), test_loader)
    print("Full private model accuracy: {}".format(accuracy))


if __name__ == "__main__":
    main()
