##############################################################################
# functional implementation of linear regression in jax
# source: https://coax.readthedocs.io/en/latest/examples/linear_regression/jax.html
##############################################################################


import jax
import jax.numpy as jnp
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# create our dataset
X, y = make_regression(n_features=3)
X, X_test, y, y_test = train_test_split(X, y)


# model weights
params = {
    'w': jnp.ones(X.shape[1:]),
    'b': 0.
}


def forward(params, X):
    return jnp.dot(X, params['w']) + params['b']


def loss_fn(params, X, y):
    err = forward(params, X) - y
    return jnp.mean(jnp.square(err))  # mse


grad_fn = jax.grad(loss_fn)


def update(params, grads):
    print("Gradietns: ", grads)
    # print(grads.shape)
    return jax.tree_map(lambda p, g: p - 0.05 * g, params, grads)


# the main training loop
for ep in range(50):
    loss = loss_fn(params, X_test, y_test)
    print("Loss details:")
    print(type(loss))
    print(loss)
    print(f"Epochs: {ep+1} Loss: {loss}")

    grads = grad_fn(params, X, y)
    print("Gradient details: ", grads)
    params = update(params, grads)
    break