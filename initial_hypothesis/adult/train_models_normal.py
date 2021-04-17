import os
import argparse
import torch
import json
import random
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from opacus import PrivacyEngine
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import outlier_indices
from preprocess import read_data, preprocess
from adult import AdultModel, train, test, train_and_save_private_model, save_model, load_model, absolute_model_path


def main():
    parser = argparse.ArgumentParser(description='Adult Income Prediction')
    parser.add_argument('--train_all', action='store_true',
                        help='whether or not to train all candidate models')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_models', type=int, default=3, help='number of models to train for each D')
    parser.add_argument('--learning_rate', type=float, default=0.00025, help='learning rate')
    parser.add_argument('--noise_multiplier', type=float, default=1.3, help='noise multiplier')
    parser.add_argument('--delta', type=float, default=0.00001, help='delta')
    args = parser.parse_args()

    start_time = datetime.now()

    # Check whether torch can use cuda
    print("Torch is available: {}".format(torch.cuda.is_available()))
    curr_device = torch.cuda.current_device()
    print("torch.cuda.current_device(): {}".format(curr_device))
    torch.cuda.device(curr_device)
    print("torch.cuda.device_count(): {}".format(torch.cuda.device_count()))
    print("torch.cuda.get_device_name(): {}".format(torch.cuda.get_device_name(curr_device)))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create new model directory to save trained models.
    batch_number = len(os.listdir(absolute_model_path))
    print('batch number: {}'.format(batch_number))
    try:
        os.makedirs('{}/batch-{}'.format(absolute_model_path, batch_number))
        print("Created directory.")
    except FileExistsError:
        print('error creating file :( current path: {}'.format(Path.cwd()))
        pass

    # Define hyperparameters
    test_accuracy = []
    train_loss = []
    weight_decay = 0

    loss_fn = nn.CrossEntropyLoss()
    test_loss_fn = nn.CrossEntropyLoss(reduction='sum')

    # Get data
    train_data_df = read_data("data/adult.data")
    test_data_df = read_data("data/adult.test")
    full_data_df = train_data_df.append(test_data_df)
    print("Size of full dataset: {}".format(full_data_df.shape))

    x_data_df, y_data_df = preprocess(full_data_df)
    indices = np.arange(x_data_df.shape[0])
    num_features = x_data_df.shape[1]

    # Get D and D' points. **** MODIFY THIS TO CHANGE D' ****
    d_points_to_train = np.arange(len(x_data_df))  # Size of adult dataset
    d_points_to_train = np.delete(d_points_to_train, outlier_indices)
    d_points_to_train = random.sample(list(d_points_to_train), len(outlier_indices))  # Train same number of normal models

    # Split data
    df_X_train, df_X_test, df_y_train, df_y_test, idx_train, idx_test = train_test_split(x_data_df, y_data_df, indices, test_size=0.20, random_state=42)

    # Check that indices line up
    assert(np.all((df_X_train.values == np.take(x_data_df.values, idx_train, axis=0))))

    # Normalise data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_X_train.values)
    X_test = scaler.transform(df_X_test.values)

    train_inputs = torch.from_numpy(X_train).to(torch.float64)
    train_targets = torch.from_numpy(df_y_train.values)

    test_inputs = torch.from_numpy(X_test).to(torch.float64)
    test_targets = torch.from_numpy(df_y_test.values)

    train_ds = TensorDataset(train_inputs, train_targets)
    val_ds = TensorDataset(test_inputs, test_targets)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, args.batch_size, pin_memory=True)

    # Parameters
    input_size = x_data_df.shape[1]
    num_classes = 2
    test_accuracy = []
    train_loss = []
    weight_decay = 0

    # Surrogate loss used for training
    loss_fn = nn.CrossEntropyLoss()
    test_loss_fn = nn.CrossEntropyLoss(reduction='sum')


    # Train non-private model
    model = AdultModel(input_size, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=weight_decay)

    print('Training non-private model...')
    start_time = datetime.now()
    _ = train(model, loss_fn, optimizer, args.epochs, train_loader, False)
    end_time = datetime.now()
    print('Non-private model training on ' + str(args.epochs) + ' epochs done in ', str(end_time-start_time),' seconds')
   

    # Train main private model:
    losses, epsilon, delta, best_alpha = (-1,-1,-1,-1)
    for j in range(args.num_models):
        losses, epsilon, delta, best_alpha = train_and_save_private_model(-1, j, train_loader, loss_fn, args.epochs, args.batch_size, args.learning_rate, args.noise_multiplier, args.delta, batch_number, num_features, num_classes)
       
    # Evaluate models
    loss, accuracy = test(model, val_loader, test_loss_fn)
    print("Original model accuracy: {}".format(accuracy))

    loss, accuracy = test(load_model(-1, 0, batch_number, num_features, num_classes), val_loader, test_loss_fn)
    print("Full private model accuracy: {}".format(accuracy))

    # Train leave-one-out models
    if args.train_all:
        for i in d_points_to_train:
            if i in idx_test:
                continue

            assert(i in idx_train)
            # Remove sample and label at original index i
            idx_train_prime = idx_train[idx_train != i]
            X_train_prime = np.take(x_data_df.values, idx_train_prime, axis=0)
            X_train_prime = scaler.fit_transform(X_train_prime)
            y_train_prime = np.take(y_data_df.values, idx_train_prime, axis=0)

            train_inputs_prime = torch.from_numpy(X_train_prime).to(torch.float64)
            train_targets_prime = torch.from_numpy(y_train_prime)

            train_ds_prime = TensorDataset(train_inputs_prime, train_targets_prime)
            train_loader_prime = DataLoader(train_ds_prime, args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

            # Train and save
            for j in range(args.num_models):
                train_and_save_private_model(i, j, train_loader_prime, loss_fn, args.epochs, args.batch_size, args.learning_rate, args.noise_multiplier, args.delta, batch_number, num_features, num_classes)

        # Evaluate leave-one-out private models
        total_accuracy = 0
        num_points = 0
        for i in d_points_to_train:
            if i in idx_test:
                continue
            model = load_model(i, 0, batch_number, num_features, num_classes)
            loss, accuracy = test(model, val_loader, test_loss_fn)
            total_accuracy += accuracy
            num_points += 1
            print("Model {} accuracy: {}".format(i, accuracy))
        
        # Write training parameters to file.
        training_info = vars(args)
        training_info['epsilon'] = epsilon
        training_info['delta'] = delta
        training_info['best_alpha'] = best_alpha
        training_info['model_accuracy'] = total_accuracy/num_points

        json_file = Path.cwd() / ("{}/batch-{}/training_info.json".format(absolute_model_path, batch_number))
        with json_file.open('w') as f:
            json.dump(training_info, f, indent="  ")

    time_elapsed = datetime.now() - start_time
    print("Training time: {}".format(time_elapsed))

if __name__ == "__main__":
    main()