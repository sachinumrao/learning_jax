import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
import flax.linen as nn
from flax.training import train_state
from jax import random, grad, value_and_grad
import jax.numpy as jnp
import optax
import wandb

# setup wandb logging
DOLOG = int(os.environ.get("logging", "0"))
if not DOLOG:
    os.environ["WANDB_MODE"]="offline"

###############################################################################
# Jax model training loop
# components:
# 1. create_train_state(): init model, init optimizer and build train_state object
# 2. apply_model(): takes input batch and runs forward pass, contains loss function definition within and returns grads and values
# 3. update_model(): takes train_state and gradients and applies update to params
# flow: for each batch i) call apply_model to get grads and loss, ii) send state and grads to update_model to get updated state
###############################################################################

# define model hyperparams
CONFIG = {
    "batch_size": 32,
    "num_epochs": 10,
    "learning_rate": 1e-4,
    "momentum": 0.9,
    "nesterov": True
}

RUN = wandb.init(project="MNIST_CNN_JAX", config=CONFIG)


# CNN model class
class CNNModel(nn.Module):
    """
    CNN based classifier for mnist dataset
    """

    @nn.compact
    def __call_(self, x):
        output_dim = 10
        x = nn.Conv(features=32, kernel_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2,2), strides=(2,2))
        x = nn.Conv(features=64, kernel_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2,2), strides=(2,2))
        # flatten the input
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=output_dim)(x)
        return x

# get dataset
def transform_fn(x):
    """
    MNIST data is PIL images, convert it to numpy arrays
    """
    return np.array(x, dtype=np.float32)


def get_mnist_dataset(batch_size=32):
    """
    function to get train and test dataloader for mnist dataset
    """
    local_dir = "~/Data/Projects/"
    train_dataset = MNIST(local_dir, train=True, download=True, transform=transform_fn)
    test_dataset = MNIST(local_dir, train=False, download=True, transform=transform_fn)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader

# three basic ops for model training
def create_train_state(random_key, config):
    """
    this function initiates model params, optimizer state and returns a train_state object
    """
    model = CNNModel()
    params = model.init(random_key, jnp.ones([1,28,28,1]))["params"]
    tx = optax.sgd(learning_rate=config["learning_rate"], momentum=config["momentum"], nesterov=config["nesterov"])
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)



def update_model(state, grads):
    """
    apply batch gradients to the model params contained in state and return updated state
    """
    return state.apply(grads=grads)


def apply_model():
    """
    this function handles the forward pass, loss and gradient calculation for batch
    """

    pass


# infra code for trnaing the model, a functional implementation of trainer
def train_model_epoch():
    pass


def eval_model():
    pass



def train_mnist():
    # get mnist data
    train_dataloader, test_dataloader = get_mnist_dataset(batch_size=CONFIG["batch_size"])

    # get train state
    pass


if __name__ == "__main__":
    train_mnist()
