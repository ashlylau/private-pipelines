"""
We will perform a multi-class classification problem based on the Iris data set, 
comprising of 4 features and 3 labels.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from opacus import PrivacyEngine

import numpy as np
import matplotlib.pyplot as plt

model_path = "models/model-"
absolute_model_path = "/Users/ashlylau/Desktop/year4/Indiv Project/private-pipelines/initial_hypothesis/iris/models/model-"

# Algorithm for hypothesis test
class PredictIris():
    def __init__(self, epsilon, x_test):
        # Ignore epsilon, this has already been calculated during model training. 
        self.x_test = x_test

    def quick_result(self, model_number):
        # Use model to predict results for x_test
        return predict(model_number, self.x_test)


class IrisModel(nn.Module):
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


def predict(model_number, x_test):
    model_number = model_number[0]
    # Randomly select model version to use to simulate algorithm randomness for the particular D'.
    num_models = len(os.listdir(absolute_model_path + str(model_number)))
    model_version = np.random.randint(num_models)
    # model_version = 1
    # print("Chosen model: {}".format(model_version))
    model = load_model(model_number, model_version)
    with torch.no_grad():
        y_pred = model.forward(x_test).argmax()
        # print("D: {}, y_pred: {}".format(model_number, y_pred))
    return y_pred.item()

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
    new_model = IrisModel()
    new_model.load_state_dict(torch.load(absolute_model_path + str(i) + "/" + str(j) + ".pt"))
    # Call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference. 
    # Failing to do this will yield inconsistent inference results.
    new_model.eval()
    return new_model

def train_and_save_private_model(i, j, train_loader, criterion, epochs, batch_size):
    priv_model = IrisModel()
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


