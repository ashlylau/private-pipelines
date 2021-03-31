import os
import argparse
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from keras.utils import to_categorical
from opacus import PrivacyEngine
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import outlier_indices
from iris import IrisModel, train, test, train_and_save_private_model, save_model, load_model, absolute_model_path

def main():
    parser = argparse.ArgumentParser(description='Iris Prediction')
    parser.add_argument('--train_all', action='store_true',
                        help='whether or not to train all candidate models')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--num_models', type=int, default=3, help='number of models to train for each D')
    parser.add_argument('--learning_rate', type=float, default=0.15, help='learning rate')
    parser.add_argument('--noise_multiplier', type=float, default=1.3, help='noise multiplier')
    parser.add_argument('--delta', type=float, default=0.01, help='delta')
    args = parser.parse_args()

    # Create new model directory to save trained models.
    batch_number = len(os.listdir(absolute_model_path))
    print('batch number: {}'.format(batch_number))
    try:
        os.makedirs('models/batch-{}'.format(batch_number))
    except FileExistsError:
        pass

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

    # Get D and D' points.
    d_points_to_train = np.arange(150)  # Size of iris dataset
    d_points_to_train = np.delete(d_points_to_train, outlier_indices)  # Train normal models
    # d_points_to_train = outlier_indices  # Train outlier models.

    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(x_data, y_data, indices, test_size=0.2, random_state=42)

    # Check that indices line up
    assert(np.all((x_train == np.take(x_data, idx_train, axis=0))))

    train_loader = torch.utils.data.DataLoader(
        TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train)), batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test)), batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    # Train main private model
    losses, epsilon, delta, best_alpha = (-1,-1,-1,-1)
    for j in range(args.num_models):
        losses, epsilon, delta, best_alpha = train_and_save_private_model(-1, j, train_loader, criterion, epochs, batch_size, args.learning_rate, args.noise_multiplier, args.delta, batch_number)
        
    # Train non-private model
    non_private_model = IrisModel()
    optimizer = torch.optim.Adam(non_private_model.parameters(), lr=args.learning_rate)
    _ = train(non_private_model, criterion, optimizer, epochs, train_loader, False)

    # Evaluate models
    accuracy = test(load_model(-1, 0, batch_number), test_loader)
    print("Full private model accuracy: {}".format(accuracy))

    accuracy = test(non_private_model, test_loader)
    print("Original model accuracy: {}".format(accuracy))

    # Train leave-one-out private models
    if args.train_all:
        for i in d_points_to_train:
            if i in idx_test:  # We only remove from the train set.
                continue
           
            assert(i in idx_train)
            # Remove sample and label at original index i
            idx_train_prime = idx_train[idx_train != i]
            x_train_prime = np.take(x_data, idx_train_prime, axis=0)
            y_train_prime = np.take(y_data, idx_train_prime, axis=0)

            train_loader_prime = torch.utils.data.DataLoader(
                TensorDataset(torch.Tensor(x_train_prime), torch.Tensor(y_train_prime)), batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
            
            # Train and save
            for j in range(args.num_models):  # We train multiple versions of the model to introduce randomness
                train_and_save_private_model(i, j, train_loader_prime, criterion, epochs, batch_size, args.learning_rate, args.noise_multiplier, args.delta, batch_number)

        # Evaluate leave-one-out private models
        total_accuracy = 0
        num_points = 0
        for i in d_points_to_train:
            if i in idx_test:
                continue
            model = load_model(i, 0, batch_number)
            accuracy = test(model, test_loader)
            total_accuracy += accuracy
            num_points += 1
            print("Model {} accuracy: {}".format(i, accuracy))
        
        # Write training parameters to file.
        training_info = vars(args)
        training_info['epsilon'] = epsilon
        training_info['delta'] = delta
        training_info['best_alpha'] = best_alpha
        training_info['model_accuracy'] = total_accuracy/num_points

        json_file = Path.cwd() / f'models/batch-{batch_number}/training_info.json'
        with json_file.open('w') as f:
            json.dump(training_info, f, indent="  ")

if __name__ == "__main__":
    main()
