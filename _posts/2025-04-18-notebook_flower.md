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


### <font color='orange'>Flower: A Friendly Federated Learning Research Framework on IMDB Data</font>.


We naturally first need to import torch and torchvision and loading **<font color='black'>the MNIST</font>** dataset

