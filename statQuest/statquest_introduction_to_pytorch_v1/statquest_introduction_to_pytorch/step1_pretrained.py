#!/c/tools/anaconda3/python
# coding: utf-8

# # The StatQuest Introduction to PyTorch!!!
# ## Sponsored by...
# [<img src="./images/PytorchLightningLogo@4x.png" alt="Lightning" style="width: 400px;">](https://www.pytorchlightning.ai/)
# &nbsp;&nbsp;&nbsp;&nbsp; <img src="./images/VerticalRule@4x.png" style="height: 100px;"> &nbsp;&nbsp;&nbsp;&nbsp;
# [<img src="./images/Grid_Logo@4x.png" alt="Grid" style="width: 400px;">](https://www.grid.ai/)
# 
# Copyright 2022, Joshua Starmer

# ----
# **NOTE:** This tutorial is from StatQuest's **[A Gentle Introduction to PyTorch]()**.
# 
# In this tutorial, we will use **[PyTorch](https://pytorch.org/)** to create, draw the output from, and optimize the super simple **neural network** featured in  StatQuest's **[Neural Networks Part 3: ReLU in Action!!!](https://youtu.be/68BZ5f7P94E)** This simple neural network, seen below, predicts whether or not a drug dose will be effective.
# <!-- <img src="./xgboost_tree.png" alt="An XGBoost Tree" style="width: 600px;"> -->
# <img src="./images/simple_relu.001.png" alt="A simple Neural Network" style="width: 1620px;">
# 
# The training data (below) that the neural network is fit to consist of three data points for three different drug doses. Low (**0**) and high (**1**) doses do not cure a disease, so their y-axis values are both **0**. However, when the dose is **0.5**, that dose can cure the disease, and the corresponding y-axis value is **1**.
# 
# <img src="./images/training_data_500x275.png" alt="A simple Neural Network" style="width: 250px;">
# 
# Below, we see the output of the neural network for different doses, and it fits the training data well!
# 
# <img src="./images/training_data_with_bent_shape_500x275.png" alt="A simple Neural Network" style="width: 250px;">
# 
# 
# In this tutorial, you will...
# 
# - **[Build a Simple Neural Network in PyTorch](#build)**
# 
# - **[Use the Neural Network and Graph the Output](#using)**
# 
# - **[Optimize (Train) a Parameter in the Neural Network and Graph the Output](#train)**
# 
# #### NOTE:
# This tutorial assumes that you already know the basics of coding in **Python** and are familiar with the theory behind **[Neural Networks](https://youtu.be/CqOfi41LfDw)**, **[Backpropagation](https://youtu.be/IN2XmBhILt4)**, the **[ReLU Activation Function](https://youtu.be/68BZ5f7P94E)**, **[Gradient Descent](https://youtu.be/sDv4f4s2SB8)**, and **[Stochastic Gradient Descent](https://youtu.be/vMh0zPT0tLI)**. If not, check out the **'Quests** by clicking on the links for each topic.
# 
# #### ALSO NOTE:
# I strongly encourage you to play around with the code. Playing with the code is the best way to learn from it.

# -----

# # Import the modules that will do all the work
# The very first thing we need to do is load a bunch of Python modules. Python itself is just a basic programming language. These modules give us extra functionality to create a neural network, use and graph the output for various input values, and optimize the neural network's parameters.
# 
# **NOTE:** You will need **Python 3** and have at least these versions for each of the following modules: 
# - pytorch >= 1.10.1
# - matplotlib >= 3.3.4
# - seaborn >= 0.11.0 
# 
# ### If you installed **Python 3** with [Anaconda](https://www.anaconda.com/)...
# ...then you can check which versions of each package you have with the command: `conda list`. If, for example, your version of `matplotlib` is older than **3.3.4**, then the easiest thing to do is just update all of your Anaconda packages with the following command: `conda update --all`. However, if you only want to update `matplotlib`, then you can run this command: `conda install matplotlib=3.3.4`.
# 
# ### If you need to install **PyTorch**...
# ...then the easiest thing to do is follow the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/).
# 
# ### If you need to install **seaborn**...
# ...then the easiest thing to do is follow the instructions on the [seaborn website](https://seaborn.pydata.org/installing.html).

# In[ ]:


## NOTE: Even though we use the PyTorch module, we import it with the name 'torch', which was the original name.
import torch # torch provides basic functions, from setting a random seed (for reproducability) to creating tensors.
import torch.nn as nn # torch.nn allows us to create a neural network.
import torch.nn.functional as F # nn.functional give us access to the activation and loss functions.
from torch.optim import SGD # optim contains many optimizers. Here, we're using SGD, stochastic gradient descent.

import matplotlib.pyplot as plt ## matplotlib allows us to draw graphs.
import seaborn as sns ## seaborn makes it easier to draw nice-looking graphs.


# -----

# <a id="build"></a>
# # Build a Simple Neural Network in PyTorch
# 
# Building a neural network in **PyTorch** means creating a new class with two methods: `__init__()` and `forward()`. The `__init__()` method defines and initializes all of the parameters that we want to use, and the `forward()` method tells **PyTorch** what should happen during a forward pass through the neural network.

# In[ ]:


## create a neural network class by creating a class that inherits from nn.Module.
class BasicNN(nn.Module):

    def __init__(self): # __init__() is the class constructor function, and we use it to initialize the weights and biases.
        
        super().__init__() # initialize an instance of the parent class, nn.Model.
        
        ## Now create the weights and biases that we need for our neural network.
        ## Each weight or bias is an nn.Parameter, which gives us the option to optimize the parameter by setting
        ## requires_grad, which is short for "requires gradient", to True. Since we don't need to optimize any of these
        ## parameters now, we set requires_grad=False.
        ##
        ## NOTE: Because our neural network is already fit to the data, we will input specific values
        ## for each weight and bias. In contrast, if we had not already fit the neural network to the data,
        ## we might start with a random initalization of the weights and biases.
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)
        
        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        self.final_bias = nn.Parameter(torch.tensor(-16.), requires_grad=False)
        
        
    def forward(self, input): ## forward() takes an input value and runs it though the neural network 
                              ## illustrated at the top of this notebook. 
        
        ## the next three lines implement the top of the neural network (using the top node in the hidden layer).
        input_to_top_relu = input * self.w00 + self.b00
        top_relu_output = F.relu(input_to_top_relu)
        scaled_top_relu_output = top_relu_output * self.w01
        
        ## the next three lines implement the bottom of the neural network (using the bottom node in the hidden layer).
        input_to_bottom_relu = input * self.w10 + self.b10
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scaled_bottom_relu_output = bottom_relu_output * self.w11
        
        ## here, we combine both the top and bottom nodes from the hidden layer with the final bias.
        input_to_final_relu = scaled_top_relu_output + scaled_bottom_relu_output + self.final_bias
        
        output = F.relu(input_to_final_relu)
    
        return output # output is the predicted effectiveness for a drug dose.


# Once we have created the class that defines the neural network, we can create an actual neural network and print out its parameters, just to make sure things are what we expect.

# In[ ]:


## create the neural network. 
model = BasicNN()

## print out the name and value for each parameter
for name, param in model.named_parameters():
    print(name, param.data)


# ## BAM!!!
# The values for each weight and bias in `BasicNN` match the values we see in the optimized neural network (below).
# <img src="./images/simple_relu.001.png" alt="A simple Neural Network" style="width: 810px;">

# -----

# <a id="using"></a>
# # Use the Neural Network and Graph the Output
# 
# Now that we have a neural network, we can use it on a variety of doses to determine which will be effective. Then we can make a graph of these data, and this graph should match the green bent shape fit to the training data that's shown at the top of this document. So, let's start by making a sequence of input doses...

# In[ ]:

print("== 1 ==\n");

## now create the different doses we want to run through the neural network.
## torch.linspace() creates the sequence of numbers between, and including, 0 and 1.
input_doses = torch.linspace(start=0, end=1, steps=11)

# now print out the doses to make sure they are what we expect...
input_doses


print("== 2 ==\n");
# Now that we have `input_doses`, let's run them through the neural network and graph the output...

## now run the different doses through the neural network.
output_values = model(input_doses)

print("== 3 ==\n");
## Now draw a graph that shows the effectiveness for each dose.
##
## First, set the style for seaborn so that the graph looks cool.
sns.set(style="whitegrid")

## create the graph (you might not see it at this point, but you will after we save it as a PDF).
sns.lineplot(x=input_doses, 
             y=output_values, 
             color='green', 
             linewidth=2.5)

print("== 4 ==\n");

## now label the y- and x-axes.
plt.ylabel('Effectiveness')
plt.xlabel('Dose')

## optionally, save the graph as a PDF.
plt.savefig('BasicNN.pdf')


# The graph shows that the neural network fits the training data. In other words, so far, we don't have any bugs in our code.
# # Double BAM!!!

