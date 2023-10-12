import torch # torch will allow us to create tensors.
import torch.nn as nn # torch.nn allows us to create a neural network.
import torch.nn.functional as F # nn.functional give us access to the activation and loss functions.
from torch.optim import SGD # optim contains many optimizers. Here, we're using SGD, stochastic gradient descent.

import lightning as L  # lightning has tons of cool tools that make neural networks easier
from torch.utils.data import TensorDataset, DataLoader  # these are needed for the training data

import matplotlib.pyplot as plt # matplotlib allows us to draw graphs.
import seaborn as sns           # seaborn makes it easier to draw nice-looking graphs.

import pprint

# from pytorch_lightning.utilities.seed import seed_everything # this is added because people on different computers were
# seed_everything(seed=42)                                     # getting different results.


## Create a neural network class by creating a class that inherits from LightningModule
class BasicLightningTrain(L.LightningModule):

    def __init__(
            self):  # __init__() is the class constructor function, and we use it to initialize the weights and biases.

        super().__init__()  # initialize an instance of the parent class, L.LightningModule.

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

        self.final_bias = nn.Parameter(torch.tensor( 0.0 ), requires_grad=True)

        self.learning_rate = 0.01 ## this is for gradient descent. NOTE: we will improve this value later, so, technically
                                  ## this is just a placeholder until then. In other words, we could put any value here
                                  ## because later we will replace it with the improved value.

    def forward(self, input):  ## forward() takes an input value and runs it though the neural network
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
        input_to_final_relu = (scaled_top_relu_output
                               + scaled_bottom_relu_output
                               + self.final_bias)

        output = F.relu(input_to_final_relu)

        return output  # output is the predicted effectiveness for a drug dose.

    def configure_optimizers(self): # this configures the optimizer we want to use for backpropagation.
        return SGD(self.parameters(), lr=self.learning_rate) # NOTE: We set the learning rate (lr) to our new variable

    def training_step(self, batch, batch_idx):  # take a step during gradient descent.

        ## NOTE: When training_step() is called it calculates the loss with the code below...
        input_i, label_i = batch  # collect input
        output_i = self.forward(input_i)  # run input through the neural network
        loss = (output_i - label_i) ** 2  ## loss = squared residual

        ##...before calling (internally and behind the scenes)...
        ## optimizer.zero_grad() # to clear gradients
        ## loss.backward() # to do the backpropagation
        ## optimizer.step() # to update the parameters
        return loss


                                                             # self.learning_rate
## create the neural network.
model = BasicLightningTrain()

## print out the name and value for each parameter
for name, param in model.named_parameters():
    print(name, param.data)

## Create the different doses we want to run through the neural network.
## torch.linspace() creates the sequence of numbers between, and including, 0 and 1.
input_doses = torch.linspace(start=0, end=1, steps=11)

# now print out the doses to make sure they are what we expect...
print("\ninput_doses:")
pprint.pprint(input_doses)

## now run the different doses through the neural network.
output_values = model(input_doses)

## Now draw a graph that shows the effectiveness for each dose.
##
## First, set the style for seaborn so that the graph looks cool.
sns.set(style="whitegrid")

## create the graph (you might not see it at this point, but you will after we save it as a PDF).
sns.lineplot(x=input_doses,
             y=output_values.detach(),
             color='red',
             linewidth=2)

## now label the y- and x-axes.
plt.ylabel('Effectiveness')
plt.xlabel('Dose')

plt.savefig('un_optimized.pdf')

print("\nLet's train:")

## create the training data for the neural network.
# inputs = torch.tensor([0., 0.5, 1.])
# labels = torch.tensor([0., 1., 0.])
## NOTE: Because we have so little data, and let's be honest, it's an unrealistically small
## amount of data, the learning rate algorithm, lr_find(), that we use in the next section has trouble.
## So, the point here is to show how to use lr_find() when you have a reasonable amount of data,
## which we fake here by making 100 copies of the inputs and labels.
inputs = torch.tensor([0., 0.5, 1.] * 100)
labels = torch.tensor([0., 1., 0.] * 100)

## If we want to use Lightning for training, then we have to pass the Trainer the data wrapped in
## something called a DataLoader. DataLoaders provide a handful of nice features including...
##   1) They can access the data in minibatches instead of all at once. In other words,
##      The DataLoader doesn't need us to load all of the data into memory first. Instead
##      it just loads what it needs in an efficient way. This is crucial for large datasets.
##   2) They can reshuffle the data every epoch to reduce model overfitting
##   3) We can easily just use a fraction of the data if we want do a quick train
dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)


## Now create a Trainer - we can use the trainer to...
##  1) Find the optimal learning rate
##  2) Train (optimize) the weights and biases in the model
## By default, the trainer will run on your system's CPU
trainer = L.Trainer(max_epochs = 40)


## Now let's find the optimal learning rate
# old fashion of 'tuner'
#lr_find_results = trainer.tuner.lr_find(model,
#                                        train_dataloaders=dataloader, # the training data
#                                        min_lr=0.001, # minimum learning rate
#                                        max_lr=1.0,   # maximum learning rate
#                                        early_stop_threshold=None) # setting this to "None" tests all 100 candidate rates
tuner = L.pytorch.tuner.tuning.Tuner( trainer )   # 'tuner' in lightning 2.0.9
lr_find_results = tuner.lr_find(model,
                                        train_dataloaders=dataloader, # the training data
                                        min_lr=0.001, # minimum learning rate
                                        max_lr=1.0,   # maximum learning rate
                                        early_stop_threshold=None) # setting this to "None" tests all 100 candidate rates

new_lr = lr_find_results.suggestion() ## suggestion() returns the best guess for the optimal learning rate

## now print out the learning rate
print(f"lr_find() suggests {new_lr:.5f} for the learning rate.")

# now set the model's learning rate to the new value
model.learning_rate = new_lr

## Now that we have an improved learning rate, we can train the model (optimize final_bias)
trainer.fit(model, train_dataloaders=dataloader)

print(model.final_bias.data)


## run the different doses through the neural network
output_values = model(input_doses)

## set the style for seaborn so that the graph looks cool.
sns.set(style="whitegrid")

## create the graph (you might not see it at this point, but you will after we save it as a PDF).
sns.lineplot(x=input_doses,
             y=output_values.detach(), ## NOTE: we call detach() because final_bias has a gradient
             color='green',
             linewidth=2)

## now label the y- and x-axes.
plt.ylabel('Effectiveness')
plt.xlabel('Dose')

plt.savefig('optimized.pdf')

print("BAM!!!")
