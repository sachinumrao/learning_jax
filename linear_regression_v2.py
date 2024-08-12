import jax
import jax.numpy as jnp
from jax import random


class LinearRegression:
    """
    A simple linear regression model implemented using JAX.
    """

    def __init__(self, key, n_features):
        """
        Initializes the model parameters.

        Args:
          key: A JAX PRNG key for random initialization.
          n_features: The number of features in the data.
        """
        self.key = key
        self.weights = random.normal(key, (n_features, 1))
        self.bias = random.normal(key, (1,))

    def predict(self, X):
        """
        Predicts the target values for a given input.

        Args:
          X: A JAX array of shape (n_samples, n_features) representing the input data.

        Returns:
          A JAX array of shape (n_samples, 1) representing the predicted target values.
        """
        return jnp.dot(X, self.weights) + self.bias

    def loss(self, X, y):
        """
        Calculates the mean squared error loss.

        Args:
          X: A JAX array of shape (n_samples, n_features) representing the input data.
          y: A JAX array of shape (n_samples, 1) representing the target values.

        Returns:
          The mean squared error loss.
        """
        predictions = self.predict(X)
        return jnp.mean(jnp.square(predictions - y))

    def update(self, X, y, learning_rate):
        """
        Updates the model parameters using gradient descent.

        Args:
          X: A JAX array of shape (n_samples, n_features) representing the input data.
          y: A JAX array of shape (n_samples, 1) representing the target values.
          learning_rate: The learning rate for gradient descent.
        """
        grads = jax.grad(self.loss)(X, y)
        self.weights = self.weights - learning_rate * grads[0]
        self.bias = self.bias - learning_rate * grads[1]

    def train(self, X, y, epochs, learning_rate):
        """
        Trains the model for a given number of epochs.

        Args:
          X: A JAX array of shape (n_samples, n_features) representing the input data.
          y: A JAX array of shape (n_samples, 1) representing the target values.
          epochs: The number of training epochs.
          learning_rate: The learning rate for gradient descent.
        """
        for epoch in range(epochs):
            self.update(X, y, learning_rate)
            loss = self.loss(X, y)
            print(f"Epoch {epoch+1}, Loss: {loss}")


# Example usage
key = random.PRNGKey(0)
n_features = 2
model = LinearRegression(key, n_features)

# Generate some sample data
X = random.normal(key, (100, n_features))
y = 2 * X[:, 0] + 3 * X[:, 1]  # + random.normal(key, (100, 1))

print("Shape of X: ", X.shape)
print("Shape of Y: ", y.shape)

# Train the model
model.train(X, y, epochs=100, learning_rate=0.01)

# Make predictions on new data
X_new = random.normal(key, (10, n_features))
predictions = model.predict(X_new)
print("Predictions:", predictions)
