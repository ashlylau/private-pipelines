"""
Adult income dataset is a binary classification problem.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from opacus import PrivacyEngine

import numpy as np
import matplotlib.pyplot as plt

model_path = "models/batch-"
absolute_model_path = "/homes/al5217/private-pipelines/initial_hypothesis/adult/models"
# absolute_model_path = "/Users/ashlylau/Desktop/year4/Indiv Project/private-pipelines/initial_hypothesis/adult/models"

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

# Algorithm for hypothesis test
class PredictAdult():
    def __init__(self, epsilon, args):
        # Ignore epsilon, this has already been calculated during model training. 
        x_test, batch_number = args
        self.x_test = x_test
        self.batch_number = batch_number

    def quick_result(self, model_number):
        # Use model to predict results for x_test
        return predict(model_number, self.x_test, self.batch_number)


class AdultModel(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.linear = torch.nn.Linear(in_size, out_size).to(torch.float64)
        
    def forward(self, xb):
        out = self.linear(xb)
        return out


def predict():
    return 0

def train(model, loss_fn, train_loader, optimizer, epoch):
    model.train()
    
    for inputs, target in train_loader:
        inputs, target = inputs.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_fn(output, target)

        loss.backward()
        optimizer.step()

def test(model, test_loader, test_loss_fn):
    model.eval()
    
    test_loss = 0
    correct = 0
    test_size = 0
    
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs, target = inputs.to(device), target.to(device)
            
            output = model(inputs)
            test_size += len(inputs)
            test_loss += test_loss_fn(output, target).item() 
            pred = output.max(1, keepdim=True)[1] 
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= test_size
    accuracy = correct / test_size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_size,
        100. * accuracy))
    
    return test_loss, accuracy
