import jax
import numpy as np


class Params:
    def __init__(self):
        self.params = {
            "a": 1.0,
            "b": 0.5,
            "c": 0.2,
            "d": 0.1
        }

p = Params()

def forward(params, x):
    a = params["a"]*x*x*x
    b = params["b"]*x*x
    c = params["c"]*x
    d = params["d"]
    return a,b,c,d

# def get_cube(params, x):
#     return params["a"]*x*x*x + params["b"]*x*x + params["c"]*x + params["d"]

def get_sum(a,b,c,d):
    return a+b+c+d

grad_fn = jax.grad(get_sum)

x = np.array([float(i) for i in range(5)])
print(x)

for i in range(5):
    print("input: ", i)
    a,b,c,d = forward(p.params, x[i])
    print("gradients: ", type(grad_fn(a,b,c,d)))