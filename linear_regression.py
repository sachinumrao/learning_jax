import jax
import jax.numpy as jnp
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


# create our dataset
def generate_dataset(n_dim):
    X, Y = make_regression(n_features=n_dim)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    return (X_train, X_test, Y_train, Y_test)


class LinearRegression:
    """Jax model class for linear regression model"""

    def __init__(self, n_dim) -> None:
        """define weights of the model"""
        weights_array = np.array([0.2, 0.3] * (n_dim[0] // 2))

        weights = jnp.array(weights_array)
        self.params = {"w": weights, "b": 0.0}

    def forward(self, X):
        """define the forwad pass of the model"""
        return jnp.dot(X, self.params["w"]) + self.params["b"]


class Trainer:
    def __init__(self, model, epochs, lr):
        self.model = model
        self.epochs = epochs
        self.lr = lr

        self.grad_fn = jax.grad(Trainer.loss_fn)

    @staticmethod
    def loss_fn(params, y_hat, y):
        """define the loss function for model"""
        err = y_hat - y
        loss = jnp.mean(jnp.square(err))
        print("Loss: ", loss)
        return loss

    def update(self, grads):
        """update model weights for given gradients"""
        print("Before Gradient: ", self.model.params)
        print("The Gradients: ", grads)
        params = jax.tree.map(lambda p, g: p - self.lr * g, self.model.params, grads)
        print("After Gradients: ", self.model.params)
        return params
        # self.model.params["w"] -= self.lr * grads["w"]
        # self.model.params["b"] -= self.lr * grads["b"]

    def train(self, X, Y):
        """implements training one step of network"""
        # run forward pass and calculate loss
        y_hat = self.model.forward(X)
        loss = Trainer.loss_fn(self.model.params, y_hat, Y)
        print("Loss fn output: ", type(loss), loss)

        # use jax autograd to get gradients for given loss
        grads = self.grad_fn(self.model.params, y_hat, Y)
        print("Gradient output: ", type(grads))
        print("Actual Gradients: ", grads)

        # update model weights with gradients
        self.model.params = self.update(grads)
        return loss

    def eval(self, X, Y):
        """evaluate model"""
        y_hat = self.model.forward(X)
        eval_loss = Trainer.loss_fn(self.model.params, y_hat, Y)
        return eval_loss

    def fit(self, X_train, Y_train, X_test, Y_test):
        """run train and eval steps for n_epochs"""
        for ep in range(self.epochs):
            train_loss = self.train(X_train, Y_train)
            eval_loss = self.eval(X_test, Y_test)
            print(f"Train Steps: {ep+1}")
            print(f"train loss: {train_loss}")
            print(f"eval loss: {eval_loss}")


def main():
    # params
    n_dim = 10
    num_epochs = 10
    lr = 3e-5

    # get dataset
    X_train, X_test, Y_train, Y_test = generate_dataset(n_dim)
    print("Dataset Generated...")
    print("Train Data: ", X_train.shape, Y_train.shape)
    print("Test Data: ", X_test.shape, Y_test.shape)
    print("n_dim: ", X_train.shape[1:])

    # init model
    model = LinearRegression(X_train.shape[1:])
    print("Model init...")

    # init trainer class
    trainer = Trainer(model, num_epochs, lr)
    print("Trainer init...")

    # start training the model
    print("Starting model training...")
    trainer.fit(X_train, Y_train, X_test, Y_test)


if __name__ == "__main__":
    main()
