---
layout: post

title: "The Future Growth of Data "

author: "Kristina"

header-mask: 0.3

mathjax: true

tags:
  - Federated learning

  - Data
---


Data is the fuel of our future. Today, data is collected in several ways enabling edge devices such as mobile phones, sensor networks or vehicles have access to the wealth of useful information. More importantly, actionable data will continuously keep us to revolutionize ideas in many domains. In the past, adding intelligence and produce powerful information into any system is provided by human beings. Today, most of intelligence can be or mostly formed by machines. Here, "machines" refers to the specific techniques and tools (algorithms) employed to gather and find patterns to give out predictions. Presently, we are in the era of where there are data located on a wide variety of locations. Integrating these types of data is highly risky as the owner and locations of where these data located are different and elsewhere. By contrast, a centralized data consists of a single data's owning only by one person or organization or country and locating at one place. Think about data in an independent and identically distributed (i.i.d.) mode. The terms of i.i.d. relates to two conditions where the characteristics for the observations or samples of one data are mutually independent and have the same distribution. So, we could not find trends in the i.i.d.' samples because statistically all data are identical to a uniformly drawn sample. Data is big and getting bigger. If we take into account one of the variables and provide all the data regarding the changing of that particular variable, that data might be in a non-i.i.d mode because the trends is changing and manage uncertainty. Apparently, with the technology’s advances, world of data is constantly changing and evolving. Dealing with this data are becoming challenges and classical centralized techniques can not accurately addressed these challenging tasks as privacy-preserving framework is the main concern. In such case, federated learning as a distributed machine learning technique withdrawn the setback of classical non-federated centralized machine learning techniques. By using a concept of federated learning, we now can have a single global model by aggregating distributed machine learning models of different client without requiring data to move from their local database. In this way, federated learning is useful for recognizing the pattern of different groups (here groups may refer to their competitor or other parties in their field), benefiting the valuable information that they do not own personally. Unlike classic non-federated frameworks that bring the data into the machine learning techniques, the federated learning works in the opposite way by pushing the machine learning techniques into the data without requiring any data shared in any way. 



# 1. Dirty Data

Most of real-world cases are in non-i.i.id mode. Dirty data is the data that is in an incorrect format or irrelevant to the context of learning. If such data is input into the machine, the results will not be as desired. This also becomes a burden for developers who need to sift through the data and find useful information to build a model.

<a href="https://idego-group.com/blog/2020/02/25/data-as-a-fuel-to-ignite-the-ai-fire/" target="_blank" rel="noopener">Sources: click here</a>


# 2. Federated Learning

Federated learning (FL) is a machine learning setting where many clients (e.g., mobile devices or whole organizations) collaboratively train a model under the orchestration of a central server (e.g., service provider), while keeping the training data decentralized. Federated learning is generally used in supervised manner where data labels are readily availabe. 

<a href="https://blog.ml.cmu.edu/2019/11/12/federated-learning-challenges-methods-and-future-directions/" target="_blank" rel="noopener">Sources: click here</a>


# 3. Potential Application

Potential applications of federated learning may include tasks such as learning the activities of mobile phone users, adapting to pedestrian behavior in autonomous vehicles, or predicting health events like heart attack risk from wearable devices.

Three canonical applications of federated learning is provided below.

1. _Learning over smart phones_. By jointly learning user behavior across a large pool of mobile ophones, statistical models can power applications such as next-word prediction, face detection, and voice recognition. However, users may not be willing to physicslly transfer their data to a central server in order to protect their personal privacy or to save the limited bandwith/battery power of their phones. _Federated learning_ has the potential to enable predictive features on smart phones without diminishing the user experience or leaking private information. 

2. _Learning across organizations_. Organizations such as _hospitals_ can also be viewed as remote 'devices' that contain a multitude of patient data for predictive healthcare. However, hospitals operate under strict privacy practises, and may face legal, administrative, or ethical constraints that require data to remain local. Federated learning is a promising solution for these applications, as it can reduce strain on the network and enable private learning between various devices/organizations. 

3. _Learning over the Internet of Things (IoT)_. Modern Internet of Things networks, such as wearable devices, autonomous vehicles, or smart homes, may contain numerous sensors that allow them to collect, react, and adapt to incoming data in real-time. For example, a fleet of autonomous vehicles may require an up-to-date model of traffic, construction, or pedestrian behavior to safely operate; however, building aggregate models in these scenarios may be difficult due to the private nature of the data and the limited connectivity of each device. Federated learning methods can help train models that efficiently adapt to changes in these systems, while maintaining user privacy. 




The canonical federated learning problem involves learning a _single_, _global_ statistical model from data stored on tens to potentially millions of remote devices. Mathematically, FL can be formulated in the following way:

$$
\min\limits_w F(w), ~ \text{where} ~ J(w):= \sum_{m=1}^M p_m J_m (w)
\tag{1}
$$

where:
- $M$ is the total number of devices or clients
- $J_m$ is the local objective function for the $m$-th device or client
- $p_m$ specifies the relative impact of each device with $p_m \geq 0$ and $\sum_{m=1}^M p_m =1$. The relative impact of each device $p_m$ is user-defined, with two natural settings being $p_m = \frac{1}{m}$ or $p_m = \frac{n_m}{n}$, where $n$ is the total number of samples over all devices.
- $J_m(w)$ is defined as the empirical risk over local data, i.e., $J_m(w) = \frac{1}{n_m} \sum_{m=1}^M \sum_{i=1}^{n(m)} \sum_{k=1}^c \mu_{[m]ik} \|x_{[m]i} - a_{[m]k}\|^2$ 


Here we propose Federated averaging algorithm designed for federated learning of collaborative unsupervised multi-view clustering. FL proceeds in multiple rounds of communication between the central server and the clients. Our federated learning employ these three stages:
- Stage 1: A central server transmites a model back to all participating clients;
- Stage 2: The clients train that model using their own local data and send back updatade models to the cetral server;
- Stage 3: The central server aggregates the updates via averaging strategy and applies the update to the shared model.
- cycle repeats.


Intuitively, _local training_ (no FL participation) and _FedAvg_ (full FL participation) can be viewed as two ends of a **personalization spectrum** with _identical_ privacy costs. More concretely, each local update step takes the following form (mean-regularized multi-task learning (MR-MTL)):

$$
w_m^{t+1} = w_m^{(t)} - \eta \bigg( {\color{yellow}\underbrace{g_m^{(t)}}_{\text{private data gradient}}} + {\color{yellow} \overbrace{\lambda (w_m^{(t)} - \bar{w}^{(t)})}^{\text{mean-regularization}}} \bigg)
\tag{2}
$$

where the hyperparameter $\lambda$ serves as a smooth knob between local training and FedAvg: $\lambda=0$ recovers local training, and a larger $lambda$ forces the personalized models to be closer to each other (intuitively, _"Federate More"_).


Tolerant to dropped devices or clients: An alternative approach involves actively selecting participating devices at each round.

