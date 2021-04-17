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
absolute_model_path = "/vol/al5217-tmp/adult/models"
absolute_data_path = "/homes/al5217/private-pipelines/initial_hypothesis/adult/data"
# absolute_model_path = "/homes/al5217/private-pipelines/initial_hypothesis/adult/models"
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


# Predict class for x_test using model-model_number
def predict(model_number, x_test, batch_number):
    model_number = model_number[0]
    # Randomly select model version to use to simulate algorithm randomness for the particular D'.
    num_models = len(os.listdir(absolute_model_path + "/batch-" + str(batch_number) + '/model-' + str(model_number)))
    model_version = np.random.randint(num_models)
    # print("Chosen model: {}".format(model_version))
    model = load_model(model_number, model_version, batch_number, 97, 2)
    x_test = x_test.to(device)
    with torch.no_grad():
        y_pred = model.forward(x_test).argmax()
        print(y_pred)
    # print("Model number: {}, prediction for x_test {} = {}".format(model_number, x_test, y_pred.item()))
    return y_pred.item()

def train(model, loss_fn, optimizer, epochs, train_loader, train_private=True, delta=0.01):
    losses = []
    for _ in range(epochs):
        for inputs, target in train_loader:
            inputs, target = inputs.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_fn(output, target)
            losses.append(loss)
            # print(f'epoch: {i:2}  loss: {loss.item():10.8f}')

            loss.backward()
            optimizer.step()

    # Print privacy budget spent.
    epsilon, best_alpha = (-1,-1)
    if train_private:
        epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(delta)
        print(
            f"Train Epoch: {epochs} \t"
            f"(ε = {epsilon:.2f}, δ = {delta}) for α = {best_alpha}"
        )

    return losses, epsilon, delta, best_alpha

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

# i : index of data left out; j : trained iteration
def save_model(model, i, j, batch_number):
    # Create folder if it doesn't exist yet.
    try:
        os.makedirs("{}/batch-{}/model-{}".format(absolute_model_path, batch_number, i))
    except FileExistsError:
        pass
    torch.save(model.state_dict(), "{}/batch-{}/model-{}/{}.pt".format(absolute_model_path, batch_number, i, j))

def load_model(i, j, batch_number, num_features, num_classes):
    new_model = AdultModel(num_features, num_classes).to(device)
    new_model.load_state_dict(torch.load(absolute_model_path + "/batch-" + str(batch_number) + "/model-" + str(i) + "/" + str(j) + ".pt"))
    # Call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference. 
    # Failing to do this will yield inconsistent inference results.
    new_model.eval()
    return new_model

def train_and_save_private_model(i, j, train_loader, criterion, epochs, batch_size, learning_rate, noise_multiplier, delta, batch_number, num_features, num_classes):
    priv_model = AdultModel(num_features, num_classes).to(device)
    priv_optimizer = torch.optim.Adam(priv_model.parameters(), lr=learning_rate)
    privacy_engine = PrivacyEngine(
        priv_model,
        batch_size=batch_size,
        sample_size=len(train_loader.dataset),
        alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
        noise_multiplier=noise_multiplier,
        max_grad_norm=1.0
    )
    privacy_engine.attach(priv_optimizer)
    print("Training model {}:".format(i))
    losses, epsilon, delta, best_alpha = train(priv_model, criterion, priv_optimizer, epochs, train_loader, True, delta)
    
    # # Plot loss
    # plt.plot(range(len(losses)), losses)
    # plt.ylabel('Loss')
    # plt.xlabel('epoch')
    # plt.title("training_loss-" + str(i))
    # plt.savefig("training_loss/training_loss-" + str(i) + ".png")
    # plt.clf()

    # Save model
    save_model(priv_model, i, j, batch_number)
    return losses, epsilon, delta, best_alpha
