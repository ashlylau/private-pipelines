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
from adult import AdultModel, train, test, absolute_model_path

def main():
    parser = argparse.ArgumentParser(description='Adult Income Prediction')
    parser.add_argument('--train_all', action='store_true',
                        help='whether or not to train all candidate models')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_models', type=int, default=3, help='number of models to train for each D')
    parser.add_argument('--learning_rate', type=float, default=0.00025, help='learning rate')
    parser.add_argument('--noise_multiplier', type=float, default=1.3, help='noise multiplier')
    parser.add_argument('--delta', type=float, default=0.01, help='delta')
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
        os.makedirs('models/batch-{}'.format(batch_number))
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
    # TODO

    df_X_train, df_X_test, df_y_train, df_y_test, idx_train, idx_test = train_test_split(x_data_df, y_data_df, indices, test_size=0.20, random_state=42)

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

    print('Training non-priavte model...')
    start_time = time.time()

    for epoch in range(1, args.epochs+1):
        print('Epoch ', epoch, ':')
        train(model, loss_fn, train_loader, optimizer, epoch)
        loss, acc = test(model, val_loader, test_loss_fn)
        
        # save results every epoch
        test_accuracy.append(acc)
        train_loss.append(loss)
 
    end_time = time.time()
    print('Non-private model training on ' + str(args.epochs) + ' epochs done in ', str(end_time-start_time),' seconds')
   

    # Train private model:
    private_model = AdultModel(num_features, 2).to(device)
    priv_optimizer = torch.optim.Adam(private_model.parameters(), lr=args.learning_rate, weight_decay=weight_decay)
    privacy_engine = PrivacyEngine(
        private_model,
        batch_size=args.batch_size,
        sample_size=len(train_loader.dataset),
        alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=1.0
    )
    privacy_engine.attach(priv_optimizer)
    print('Training private model...')
    start_time = time.time()

    for epoch in range(1, args.epochs+1):
        print('Epoch ', epoch, ':')
        train(private_model, loss_fn, train_loader, priv_optimizer, epoch)
        loss, acc = test(private_model, val_loader, test_loss_fn)
        
        # save results every epoch
        test_accuracy.append(acc)
        train_loss.append(loss)
        
    end_time = time.time()
    print('Private model training on ' + str(args.epochs) + ' epochs done in ', str(end_time-start_time),' seconds')
 
    # TODO: leave one out models.

    # Evaluate models
    loss, accuracy = test(model, val_loader, test_loss_fn)
    print("Original model accuracy: {}".format(accuracy))

    loss, accuracy = test(private_model, val_loader, test_loss_fn)
    print("Full private model accuracy: {}".format(accuracy))


if __name__ == "__main__":
    main()