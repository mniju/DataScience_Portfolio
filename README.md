# Data Science Portfolio

My collection of Datascience Projects completed as part of MOOCS and Self study .
## Machine Learning
### Classification 
1. [Wine Quality Classification](https://github.com/mniju/WineQualityAnalysis/blob/master/WineQuality.ipynb): Analysis of the [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Wine+Quality), Visualize and then try different ML algorithms on that - Decision Trees, SVM , Clusterin
### RecSys

1. [Collborative Filtering](https://github.com/mniju/Recsys/blob/main/MatrixFactorization/Movie%20Recommendation.ipynb) :  Used the Matrix Factorization to design a Movie recommender system .I used the small Movie lens dataset and used different similarity checks for comparision. The related  medium post i wrote on the analysis can be found [here](https://medium.com/@niju.nicholas/collaborative-filtering-with-matrix-factorization-e5779f7fba74) .

### Clustering

1. K Means from Scratch :Developed K Means Clustering Algorithm from scratch and applied it on the Chocolate Ratings dataset in [kaggle](https://www.kaggle.com/datasets/rtatman/chocolate-bar-ratings)  and discussed it in my medium post [here](https://medium.com/@niju.nicholas/k-means-clustering-from-scratch-with-manual-similarity-measure-89620e64541) .

## Deep Learning

1. [Behaviour Cloning](https://github.com/mniju/CarND-Behavioral-Cloning-P3): Steer a Car in the Simulator by training a model with Convolutional Neural Networ based on the [nvidia paper](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/) *End-to-End Deep Learning for Self-Driving Cars*

2. [Vehicle Detection](https://github.com/mniju/Vehicle-Detection) : Detect the Vehicles in a Video with SVM Classifier and draw a bounding box for the vehicles detected.

3. [Semantic Segmentation](https://github.com/mniju/CarND-Semantic-Segmentation): Train a network to classify each pixel in the Image to two classes -Road/Not Road, using the Encoder-Decoder Model.


## Reinforcement Learning

1. [Q Learning and SARSA](https://github.com/mniju/Practical_RL-Yandex/blob/master/week03_model_free/homework.ipynb) : Run both Q learnng and SARSA agents on a Cliff walking enviornemnt and compare the results of the actions between two algorithms.
   
2. [Deep Q Learning to Play Atari Pong](https://github.com/mniju/Practical_RL-Yandex/blob/master/week04_approx_rl/homework_pytorch_main.ipynb) : This is a implementation of the paper [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf) . This uses a deep Neural network with CNN to learn from the Images.A replay buffer is used to collect the agent experience and an additional  target network is used to prevent action bias.
   
3. [Distributional RL - C51](https://github.com/mniju/Practical_RL-Yandex/blob/master/week04_approx_rl/C51-Atari.ipynb): Implementation of the distributional RL paper [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/pdf/1707.06887.pdf) well known as  **C51** for the Atari Pong game .
   
4. [Policy Gradients- REINFORCE](https://github.com/mniju/Practical_RL-Yandex/blob/master/week06_policy_based/reinforce_pytorch.ipynb) : Implementation of RL REINFORCE Algorithm based on learning Policy Gradients wherein the policy is learnt directy rather than learning the Q Values and then making policy decisions based on the learnt Q values.
   
5. [Proximal Policy Gradient PPO](https://github.com/mniju/Practical_RL-Yandex/blob/master/week09_policy_II/ppo.ipynb): Implemented PPO algorithm which belongs to the family of policy gradients ,from  [this paper](https://arxiv.org/abs/1707.06347) on a continuous Gym   Enviornment  Half Cheetah.
   
6. Batch Constraint Q Learning: This is an off policy Reinforcememt Learning .Implemented the [BCQ Paper](https://arxiv.org/abs/1812.02900) and Ran experiments with different Gym enviormments and tracked them in [Weights and Biases](https://wandb.ai/niju/BCQ_ant-bullet-medium-v0/runs/2tb5tsak)
