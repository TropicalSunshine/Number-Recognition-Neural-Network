#standard python library 
import random

from vector    import vector
from networkm import sigmoid

class node:
    def __init__(self,i, weights, bias = 1):
        self.inputs = vector(i)
        self.w = vector(weights)
        self.b = 1

    def take_inputs(self,i):
        if len(i) != len(self.w):
            raise Exception("Number of inputs do not match number of weights")
        self.a = i


    def init_weights(self, N):
        """creates the initial weights based on the amount of inputs 
            intiailized between a random float value -1 and 1"""

        if len(self.w) != 0:
            raise AttributeError("node must not have any weights must be empty")
        for i in range(N):
            self.w.add(random.uniform(-1,1))

    def set_bias(self, b):
        self.bias = b
    
    def get_bias(self):
        return self.bias

    def set_weights(self, index, value):
        "weight must be an array "
        if index < len(self.w) or index > len(self.w) - 1:
            raise IndexError
        self.w[index] = value
    
    def get_weights(self):
        return self.w

    def activate(self):
        if len(self.a) != len(self.w):
            raise AttributeError("could not activate difference in w and a")
        return sigmoid((self.a * self.w) + self.b)
        
            



    
