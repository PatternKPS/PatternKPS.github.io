---
layout: post
title: "Flower"
date: 2024-04-19 07:00
author: "Kristina"
mathjax: true
tags:
  - Federated learning
  - Flower
---


## <font color='red'>F</font><font color='orange'>L</font><font color='magenta'>O</font><font color='yellow'>W</font><font color='green'>E</font>R

<font color='black'> The idea behind Federated Learning is to train a model between multiple clients and a server without having to share any data. This is done by letting each client train the model locally on its data and send its parameters back to the server, which then aggregates all the clients’ parameters together using a predefined strategy. This process is made very simple by</font> <font color='green'>using the Flower framework &#128512;.</font>


### <font color='orange'>Flower: A Friendly Federated Learning Research Framework on IMDB Data.</font>


We naturally first need to import torch and torchvision and loading **<font color='black'>the MNIST</font>** dataset

```ts
import flwr as fl
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "distilbert-base-uncased"
from torchvision.transforms import ToTensor, Normalize, Compose
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from flwr.common import Metrics
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

print("flwr", fl.__version__)
print("numpy", np.__version__)
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)

DEVICE = torch.device("cpu")
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on {DEVICE}")
```

```ts
model = Net(num_classes=10)
num_parameters = sum(value.numel() for value in model.state_dict().values())
print(f"{num_parameters = }")
```

The training loop

```ts
def train(net, trainloader, optimizer, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
    return net


def test(net, testloader):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy

def run_centralised(epochs: int, lr: float, momentum: float = 0.9):
    """A minimal (but complete) training loop"""

    # instantiate the model
    model = Net(num_classes=10)

    # define optimiser with hyperparameters supplied
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # get dataset and construct a dataloaders
    trainset, testset = get_mnist()
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=128)

    # train for the specified number of epochs
    trained_model = train(model, trainloader, optim, epochs)

    # training is completed, then evaluate model on the test set
    loss, accuracy = test(trained_model, testloader)
    print(f"{loss = }")
    print(f"{accuracy = }")
```

Create 2 partitions and extract some statistics from one partition &#128512;&#128512;&#128512;


```ts
from torch.utils.data import random_split
def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float = 0.1):
    """This function partitions the training set into N disjoint
    subsets, each will become the local dataset of a client. This
    function also subsequently partitions each traininset partition
    into train and validation. The test set is left intact and will
    be used by the central server to asses the performance of the
    global model."""

    # get the MNIST dataset
    trainset, testset = get_mnist()

    # split trainset into `num_partitions` trainsets
    num_images = len(trainset) // num_partitions

    partition_len = [num_images] * num_partitions

    trainsets = random_split(
        trainset, partition_len, torch.Generator().manual_seed(2023)
    )

    # create dataloaders with train+val support
    trainloaders = []
    valloaders = []
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(
            trainset_, [num_train, num_val], torch.Generator().manual_seed(2023)
        )

        trainloaders.append(
            DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2)
        )
        valloaders.append(
            DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2)
        )
    # create dataloader for the test set
    testloader = DataLoader(testset, batch_size=128)

    return trainloaders, valloaders, testloader
```


Create 100 partitions and extract some statistics from one partition &#128512;&#128512;&#128512;


```ts
import matplotlib.pyplot as plt

trainloaders, valloaders, testloader = prepare_dataset(
    num_partitions=100, batch_size=32
)

# first partition
train_partition = trainloaders[0].dataset

# count data points
partition_indices = train_partition.indices
print(f"number of images: {len(partition_indices)}")

# visualise histogram
plt.hist(train_partition.dataset.dataset.targets[partition_indices], bins=10, color = "red", ec="green")

plt.grid()
plt.xticks(range(10))
plt.xlabel("Label")
plt.ylabel("Number of images")
plt.title("Class labels distribution for MNIST")
```


### <font color='orange'>Defining a Flower Client</font>

We can think of a client in FL as an entity that owns some data and trains a model using this data. The caveat is that the model is being trained collaboratively in Federation by multiple clients (sometimes up to hundreds of thousands) and, in most instances of FL, is sent by a central server.

A Flower Client is a simple Python class with four distinct methods:

- <span style='color:  #000000; font-family: monospace; background-color: #40E0D0;'>fit()</span>: With this method, the client does on-device training for a number of epochs using its own data. At the end, the resulting model is sent back to the server for aggregation.

- <span style='color:  #000000; font-family: monospace; background-color: #40E0D0;'>evaluate()</span>: With this method, the server can evaluate the performance of the global model on the local validation set of a client. This can be used for instance when there is no centralised dataset on the server for validation/test. Also, this method can be use to asses the degree of personalisation of the model being federated.

- <span style='color:  #000000; font-family: monospace; background-color: #40E0D0;'>set_parameters()</span>: This method takes the parameters sent by the server and uses them to initialise the parameters of the local model that is ML framework specific (e.g. TF, Pytorch, etc).

- <span style='color:  #000000; font-family: monospace; background-color: #40E0D0;'>get_parameters()</span>: It extract the parameters from the local model and transforms them into a list of NumPy arrays. This ML framework-agnostic representation of the model will be sent to the server.

Start by importing Flower! &#128512;&#128512;&#128512;

```ts
import flwr as fl
import torch

DEVICE = torch.device("cpu")
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on {DEVICE}")
```

Then we need to define our Flower Client class &#128512;&#128512;&#128512;

```ts
from collections import OrderedDict
from typing import Dict, Tuple

import torch
from flwr.common import NDArrays, Scalar


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, vallodaer) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = vallodaer
        self.model = Net(num_classes=10)

    def set_parameters(self, parameters):
        """With the model parameters received from the server,
        overwrite the uninitialise model in this class with them."""

        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        # now replace the parameters
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract all model parameters and convert them to a list of
        NumPy arrays. The server doesn't work with PyTorch/TF/etc."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """This method train the model using the parameters sent by the
        server on the dataset of this client. At then end, the parameters
        of the locally trained model are communicated back to the server"""

        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)

        # Define the optimizer -------------------------------------------------------------- Essentially the same as in the centralised example above
        optim = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

        # do local training  -------------------------------------------------------------- Essentially the same as in the centralised example above (but now using the client's data instead of the whole dataset)
        train(self.model, self.trainloader, optim, epochs=1)

        # return the model parameters to the server as well as extra info (number of training examples in this case)
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Evaluate the model sent by the server on this client's
        local validation set. Then return performance metrics."""

        self.set_parameters(parameters)
        loss, accuracy = test(
            self.model, self.valloader
        )  # <-------------------------- calls the `test` function, just what we did in the centralised setting (but this time using the client's local validation set)
        # send statistics back to the server
        return float(loss), len(self.valloader), {"accuracy": accuracy}
```

Spend a few minutes to inspect the <span style='color:  #000000; font-family: monospace; background-color: #40E0D0;'>FlowerClient</span> class above. Please ask questions if there is something unclear !


Then keen-eyed among you might have realised that if we were to fuse the client's <span style='color:  #000000; font-family: monospace; background-color: #40E0D0;'>fit()</span> and <span style='color:  #000000; font-family: monospace; background-color: #40E0D0;'>evaluate()</span> methods, we'll end up with essentially the same as in the <span style='color:  #000000; font-family: monospace; background-color: #40E0D0;'>run_centralised()</span> function we used in the Centralised Training part of this tutorial. And it is true!! In Federated Learning, the way clients perform local training makes use of the same principles as more traditional centralised setup. The key difference is that the dataset now is much smaller and it's never *"seen"* by the entity running the FL workload (i.e. the central server).


Talking about the central server... we should define what strategy we want to make use of so the updated models sent from the clients back to the server at the end of the <span style='color:  #000000; font-family: monospace; background-color: #40E0D0;'>fit()</span> method are aggregate.
