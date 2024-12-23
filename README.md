# Data Science Portfolio

My collection of Datascience Projects completed as part of MOOCS and Self study .
## Machine Learning
### AI assisted  Camera Lens Alignment

In camera manufacturing process, we implemented **AI Assisted Active Alignment**, a machine learning-based process to precisely glue the camera base to the lens. Cameras have two main parts: Firstly, the image sensor with electronics and secondly the lens. While smartphone lenses dynamically adjust focus during use, automotive cameras like reverse and side view cameras require their lenses to be glued in optimal alignment during manufacturing.

Producing ~1,000 cameras daily, we sought to improve efficiency by reducing alignment cycle time using machine learning. Inspired by Google's 2020 paper "**Learning to Autofocus**," which used a *MobileNetV2* model to optimize smartphone lens adjustment, we developed our own approach. Using focus scores from production images, we trained a [Xgboost](https://xgboost.readthedocs.io/en/latest/index.html) model to predict lens position. This reduced alignment steps from 16 to just 3-4 fine-tuning steps.

Integrating this solution significantly enhanced production efficiency and throughput.

### Masters Thesis - Energy Aware VM Selection Policies for Data Centres.
This [Thesis](https://github.com/mniju/Masters_Thesis/blob/main/Reinforcement%20Learning%20Algorithms%20for%20Energy%20aware%20VM%20Selection%20in%20Data%20centre.pdf) aims to evaluate an approach to reduce data centre energy consumption using **reinforcement learning algorithms** to optimize the virtual machine (VM) selection process.**VM selection** is the process of selecting an VM from a overloaded host and moving it to another host.An optimized selection of VMs can lead to a few overloaded hosts and this leads to a reduction in energy usage.

Two reinforcement learning algorithms, Q-learning and SARSA were implemented in the [cloudsim toolkit](https://github.com/Cloudslab/cloudsim). Subsequent experiments employed two distinct policies—epsilon greedy and SoftMax—to determine the most effective hyperparameters, specifically the learning rate (alpha) and the discount factor (gamma). The proposed algorithm results in a energy saving of 18% compared to the Lr-Mmt approach. The results of this thesis conclude that the RL algorithm can intelligently optimize the VM selection process and thereby reducing the energy consumption in the data center.

## Deep Learning

1. [Behaviour Cloning](https://github.com/mniju/CarND-Behavioral-Cloning-P3): Built an end-to-end self-driving solution using the CARLA simulator, inspired by the [NVIDIA paper](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/) *End-to-End Deep Learning for Self-Driving Cars*. The model processes input from the center camera and outputs steering angles. During training, data augmentation techniques were applied, including flipping images, translating images to simulate random steering, and adjusting brightness. Additionally, left and right camera images were used to enhance recovery, with steering angle offsets applied as described in the NVIDIA approach.

The model architecture features a convolutional neural network with five convolutional layers (filter sizes: 5x5 and 3x3; depths: 24 to 64) followed by four fully connected layers. This implementation successfully enables the vehicle to drive autonomously around the track without veering off the road.

2. [Vehicle Detection](https://github.com/mniju/Vehicle-Detection) : Detect the Vehicles in a Video with SVM Classifier and draw a bounding box for the vehicles detected.

3. [Semantic Segmentation](https://github.com/mniju/CarND-Semantic-Segmentation): Train a network to classify each pixel in the Image to two classes -Road/Not Road, using the Encoder-Decoder Model.


## Reinforcement Learning

1. [Q Learning and SARSA](https://github.com/mniju/Practical_RL-Yandex/blob/master/week03_model_free/homework.ipynb) : Run both Q learnng and SARSA agents on a Cliff walking enviornemnt and compare the results of the actions between two algorithms.
   
2. [Deep Q Learning to Play Atari Pong](https://github.com/mniju/Practical_RL-Yandex/blob/master/week04_approx_rl/homework_pytorch_main.ipynb) : This is a implementation of the paper [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf) . This uses a deep Neural network with CNN to learn from the Images.A replay buffer is used to collect the agent experience and an additional  target network is used to prevent action bias.
   
3. [Distributional RL - C51](https://github.com/mniju/Practical_RL-Yandex/blob/master/week04_approx_rl/C51-Atari.ipynb): Implementation of the distributional RL paper [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/pdf/1707.06887.pdf) well known as  **C51** for the Atari Pong game .
   
4. [Policy Gradients- REINFORCE](https://github.com/mniju/Practical_RL-Yandex/blob/master/week06_policy_based/reinforce_pytorch.ipynb) : Implementation of RL REINFORCE Algorithm based on learning Policy Gradients wherein the policy is learnt directy rather than learning the Q Values and then making policy decisions based on the learnt Q values.
   
5. [Proximal Policy Gradient PPO](https://github.com/mniju/Practical_RL-Yandex/blob/master/week09_policy_II/ppo.ipynb): Implemented PPO algorithm which belongs to the family of policy gradients ,from  [this paper](https://arxiv.org/abs/1707.06347) on a continuous Gym   Enviornment  Half Cheetah.
   
6. Batch Constraint Q Learning: This is an off policy Reinforcememt Learning .Implemented the [BCQ Paper](https://arxiv.org/abs/1812.02900) and Ran experiments with different Gym enviormments and tracked them in [Weights and Biases](https://wandb.ai/niju/BCQ_ant-bullet-medium-v0/runs/2tb5tsak)
