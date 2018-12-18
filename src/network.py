import array as Array
import node as Node
import layer as Layer

#global
E = 2.718281828459045


def sigmoid(x):
    "the sigmoid function"
    return 1.0/(1.0+(E**(-x)))


def sigmoid_prime(x):
    "derivative of the sigmoid function"
    "growth of the function"
    return sigmoid(x)*(1 - sigmoid(x))






def main():
    pass




if __name__ == "__main__":
    main()