---
layout: post
title: "Federated Learning"
date: 2024-04-19 07:00
author: "Kristina"
mathjax: true
tags:
  - Federated learning
---




### <font color='black'> The idea behind Federated Learning is to train a model between multiple clients and a server without having to share any data. This is done by letting each client train the model locally on its data and send its parameters back to the server, which then aggregates all the clients’ parameters together using a predefined strategy. This process is made very simple by</font> <font color='green'>using the Flower framework</font> &#128512;.


## <font color='red'>F</font><font color='orange'>l</font><font color='magenta'>o</font><font color='yellow'>w</font><font color='green'>e</font>r</font> - **<font color='orange'>A Friendly Federated Learning Research Framework on IMDB Data</font>**.


We naturally first need to import torch and torchvision and loading **<font color='black'>the MNIST</font>** dataset

`import flwr as fl
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
print(f"Training on {DEVICE}")`