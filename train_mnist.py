import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
import flax.linen as nn
from flax.training import train_state, orbax_utils
from orbax.checkpoint import PyTreeCheckpointer

import jax
import jax.numpy as jnp
import optax
from optax import softmax_cross_entropy
import wandb

from tqdm.auto import tqdm

# setup wandb logging
DEBUG = int(os.environ.get("DEBUG", "0"))
###############################################################################
# Jax model training loop
# components:
# 1. create_train_state(): init model, init optimizer and build train_state object
# 2. train_step(): takes input batch and runs forward pass, contains loss function definition within and returns grads and values
# 3. update_model(): takes train_state and gradients and applies update to params
# flow: for each batch in data loader:
#  i) call apply_model to get grads and loss,
# ii) send state and grads to update_model to get updated state
###############################################################################

# define model hyperparams
CONFIG = {
    "batch_size": 32,
    "num_epochs": 10,
    "learning_rate": 1e-4,
    "momentum": 0.9,
    "nesterov": True,
    "log_steps": 10,
}

RUN = wandb.init(project="MNIST_CNN_JAX", config=CONFIG)


# CNN model class
class CNNModel(nn.Module):
    """
    CNN based classifier for mnist dataset
    """

    @nn.compact
    def __call__(self, x):
        output_dim = 10
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
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

    x = np.array(x, dtype=np.float32).reshape(28, 28, 1)
    print(f"in transform_fn, returning {type(x)}...") if DEBUG else None
    return x


def collate_fn(batch):
    """
    custom collate_fn keeps input in np.array, defualt collate_fn in dataloader wraps np.array into torch.Tensor
    batch: List of tuples, each tuple contains an image and its label
    """
    imgs = []
    labels = []
    for item in batch:
        imgs.append(item[0])
        labels.append(item[1])

    imgs = np.stack(imgs)
    labels = np.array(labels)

    return imgs, labels


def get_mnist_dataset(batch_size=32):
    """
    function to get train and test dataloader for mnist dataset
    """
    local_dir = "~/Data/Projects/"
    train_dataset = MNIST(local_dir, train=True, download=True, transform=transform_fn)
    test_dataset = MNIST(local_dir, train=False, download=True, transform=transform_fn)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_dataloader, test_dataloader


# three basic ops for model training
def create_train_state(random_key, config):
    """
    this function initiates model params, optimizer state and returns a train_state object
    """
    model = CNNModel()
    params = model.init(random_key, jnp.ones([1, 28, 28, 1]))["params"]
    tx = optax.sgd(
        learning_rate=config["learning_rate"],
        momentum=config["momentum"],
        nesterov=config["nesterov"],
    )
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def update_model(state, grads):
    """
    apply batch gradients to the model params contained in state and return updated state
    """
    return state.apply(grads=grads)

@jax.jit
def train_step(state, imgs, labels):
    """
    this function handles the forward pass, loss and gradient calculation for batch
    """

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, imgs)
        ground_truth = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(softmax_cross_entropy(logits=logits, labels=ground_truth))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy


# infra code for trnaing the model, a functional implementation of trainer
def train_model_epoch(state, train_dataloader, config, epoch):
    """
    Train model for one epoch
    """
    train_loss = []
    train_acc = []
    for idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        imgs, labels = batch
        grads, loss, acc = train_step(state, imgs, labels)
        train_loss.append(loss)
        train_acc.append(acc)

        # log step info
        if (idx + 1) % config["log_steps"] == 0:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "steps": (idx + 1),
                    "train_loss": loss,
                    "train_acc": acc,
                }
            )

    train_loss = np.mean(train_loss)
    train_acc = np.mean(train_acc)

    return state, train_loss, train_acc


def eval_model(state, test_dataloader):
    """
    run model eval on test dataloader
    """
    test_loss = []
    test_acc = []

    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        # figure out forward pass without gradient calculations
        imgs, labels = batch
        _, loss, acc = train_step(state, imgs, labels)
        test_loss.append(loss)
        test_acc.append(acc)

    test_loss = np.mean(test_loss)
    test_acc = np.mean(test_acc)

    return 0, test_loss, test_acc


def train_mnist(config, model_output_path):
    # get mnist data
    train_dataloader, test_dataloader = get_mnist_dataset(
        batch_size=config["batch_size"]
    )

    # get train state
    rng = jax.random.key(0)
    rng, init_rng = jax.random.split(rng)

    state = create_train_state(init_rng, config)

    # train model
    for ep in range(config["num_epochs"]):
        print("Training epoch: ", ep + 1)
        state, train_loss, train_acc = train_model_epoch(
            state, train_dataloader, config, ep
        )
        _, test_loss, test_acc = eval_model(state, test_dataloader)
        wandb.log(
            {
                "epoch": ep + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )

    print("Finished Training, saving model...")
    ckpt = {"model": state, "config": config, "data": jnp.ones([1, 28, 28, 1])}
    save_args = orbax_utils.save_args_from_target(ckpt)
    checkpointer = PyTreeCheckpointer()
    checkpointer.save(model_output_path, ckpt, save_args=save_args)


if __name__ == "__main__":
    model_output_path = "./models/mnist_jax_v1/"
    train_mnist(CONFIG, model_output_path)

# TODO:
# - write infer_step with no grads
# - write eval_step that leverages infer_step() and then computes loss and acc
# - can we take out loss function outside of train_step function??
# - model summary for jax model

###############################################################################
# Pains in jax:
# - model checkpoitning is tedious
# - managing rng keys
# - train step coupled with loss calculations
# - how to do forward pass without grad calculations
###############################################################################
