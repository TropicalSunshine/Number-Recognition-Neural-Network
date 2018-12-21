#global
E = 2.718281828459045

def sigmoid(x):
    "the sigmoid function"
    if x < (-700):
        return 0
    return 1.0/(1.0+(E**(-x)))


def sigmoid_prime(x):
    "derivative of the sigmoid function"
    "growth of the function"
    return sigmoid(x)*(1 - sigmoid(x))


def cost(actual, expected):
    return (actual - expected)**2

def cost_prime(actual, expected):
    return 2*(actual - expected)
