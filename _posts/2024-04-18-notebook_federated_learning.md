---
layout: post
title: "Federated Learning"
date: 2024-04-18 19:55
author: "Kristina"
mathjax: true
tags:
  - Federated learning
---


### Federated Learning

AI change the future. May 11, 1997, an IBM computer called IBM **<font color='blue'>Deep Blue</font>** beat the world chess champion, Garry Kasparov after a six game match. 20 years after Deep Blue, March 19, 2016, marks the day Google DeepMind's AI program, **<font color='orange'>AlphaGo</font>**, beat the strongest Go player in the world, Lee sedol. It has long been considered a difficult challenge in the field of AI and considerably more difficult to solve than chess. Go is a complex board game that requires intuition, creative, and strategic thinking. To train AlphaGo, the machine was introduced to a number of amateur games to help develop an understandable image of what reasonable human play looked like. The strongest computer programs play human amateurs at Go has one neural network referred to as the **<font color='orange'>"policy network"</font>**, which selects the next move to play. The other neural network is referred to as the **<font color='orange'>"value network"</font>**, and it predicts the winner of the game. Those two moments proving that a computer became better than humans at Chess and AlphaGo. After 6 years, November 30, 2022, **<font color='orange'>ChatGPT</font>** which stands for *Chat Generative Pre-trained Transformer* is launched by OpenAI. ChatGPT is a chatbot based on *Large Language Model (LLM)* which enables users to ask questions (prompts) and ChatGPT will write a text for it. ChatGPT has a wide range of abilities, everything from passing MBA exam, writing poems, writing codes, solving mathematical problems, generate images/videos, creating a content, etc. According to OpenAI, ChatGPT acquired 1 million users just 5 days after it launched. We have truly witnessed the huge potential in AI and have began to expect more complex, cutting-edge AI technology in many applications.   

 As AI did revolutionize and will continously change the game. Recently, **<font color='orange'>privacy-preserving machine learning</font>** via secure **<font color='green'>multiparty computation (MPC)</font>** has emerged as **<font color='orange'>an active area of research</font>** that allows different entities to train various models on their joint data without revealing any information except the output. The next generation of AI will be built upon the core idea revolving around **"data privacy"**.

 ### Definition

 Federated learning (FL) is **<font color='green'>the future of AI</font>** which can aggregate multiple clients' model to generate a single global model with data privacy and security concerns. So, generally the federated learning is kept the client' data locally on their database and their local model updates are based on the global model passed by a centralized server. The idea of federated learning firstly invented by Google (<a href="https://arxiv.org/abs/1602.05629" target="_blank" rel="noopener">McMahan et al.,</a>) in 2016. <a href="https://arxiv.org/pdf/1610.05492.pdf" target="_blank" rel="noopener">Konečný et al.,</a> then proposed a communication efficiency to investigate the effect of **<font color='green'>deep neural networks</font>** for two different tasks on multiple clients' data such as **<font color='green'>CIFAR-10 image classification</font>** task using convolutional networks and **<font color='green'>next word prediction (Google BigQuery) using recurrent network</font>**. In short, they employ **<font color='red'>the Federated Averaging algorithm (FedAvg)</font>** to train a good model. However, their techniques only utilizing random selected clients on each round with a learning rate of $\eta$ on client'local dataset with a purpose to reduce the communication costs associated with fetching local model changes to the central aggregation server.


  ### Example

  Federated learning simply aims to `bring a machine learning technique to data, not bring the data to a certain type of machine learning technique (conventional machine learning moves the data to the computation, while federated (machine) learning moves the computation to the data)`. Specifically, it naturally enables machine learning on distributed data by moving the training to the data, instead of moving the data to the training. By moving the machine learning to the data, we can collectively train a model as a whole.

The simple explanation between central (conventional or classical) machine learning and federated machine learning:

1. In terms of **<font color='greeb'>data privacy</font>**, the `conventional machine learning(CML)` carry out the complete training process on a single server which may poses several privacy risks when the information is shared with the central cloud server. In contrast, `federated machine learning (FML)` enables participant to train local models cooperatively on local data without disclosing sensitive data to the central cloud server (ensuring no end-user-data leaves the device).

2. In terms of **<font color='green'>data distribution</font>**, `conventional machine learning(CML)` techniques assumes that the clients' data is independent and identically distributed (i.i.d). While, `federated machine learning (FML)` assumes that the clients' data is in non-independent and identically distributed (non-i.i.d.) mode as clients have different data types. 


    In the literature, based on the distribution characteristics of the data, FL is categorized into three (see: <a href="https://dl.acm.org/doi/pdf/10.1145/3298981" target="_blank" rel="noopener">Yang et al.,</a>, <a href="https://www.sciencedirect.com/science/article/pii/S0950705121000381" target="_blank" rel="noopener">Zhang et al.,</a>, and <a href="https://www.sciencedirect.com/science/article/pii/S0925231221013254" target="_blank" rel="noopener">Zhu et al.,</a>):

        - `Horizontal federated learning (HFL)`. Horizontal FL is also called homogeneous FL, which represents the scenarios in which the training data of participating clients share the same feature space but have different sample space. This means that client 1 and client 2 has the same set of features. The FedAvg is a typical HFL.

        - `Vertical federated learning (VDL)`. VDL is also referred to heterogeneous FL, in which clients' data share the same sample space but have different feature space to jointly train a global model.

        - `Federated transfer learning (FTL)`. FTL applies to the scenarios in which clients' data share the differ not only in training sample space but also in feature space.


3. In terms of **<font color='red'>continual learning</font>**, `CML model` is developed in a centralized setting using all available training data. The idea of *continual learning* is to mimic humans ability to continually acquire, fine-tune, and transfer knowledge and skills throughout their lifespan. In practice, this means supporting the baility of a model autonomously learn and adapt in production as new data comes in (we know that data is changing because of trends, or because of different actions made by the users). In CML, this works flawlessly when a central server is available to serve the forecasts. But may be took long to provide a satisfactory user experience when users expect quick responses. The `federated continual learning (FCL)` poses challenges to continual learning, such as utilizing knowledge from other clients, while preventing interference from irrelevant knowledge. This is not very realistic in FL environments where each client works independently in an asynchronous manner getting data for the different tasks in time-frames and orders totally uncorrelated with the other ones. So, continuous learning is difficult in federated environments. 


4. In terms of **<font color='red'>aggregation of datasets</font>**, the `CML` involves aggregating user data in a central location, which may violate certain nations' privacy rules and make data more vulnerable to data breaches. While `FML` models are constantly upgraded, allowing client input, and there is no need to aggregate data for continuous learning.


**<font color='orange'>Federated learning (FL) enables us to use machine learning (and other data science approaches) in areas where it wasn’t possible before</font>**. We can now train excellent medical AI models by enabling different hospitals to work together. We can solve financial fraud by training AI models on the data of different financial institutions. We can build novel privacy-enhancing applications (such as secure messaging) that have better built-in AI than their non-privacy-enhancing alternatives. And those are just a few of the examples that come to mind. As we deploy federated learning, we discover more and more areas that can suddenly be reinvented because they now have access to vast amounts of previously inaccessible data.


<a href="https://flower.dev/docs/framework/tutorial-series-what-is-federated-learning.html" target="_blank" rel="noopener">Sources: click here</a>