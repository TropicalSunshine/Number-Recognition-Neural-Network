
import gzip
from PIL import Image

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

def unzip():
    f = gzip.open("../data/t10k-labels-idx1-ubyte.gz",'rb')
    label = f.readlines()
    labels = []
    for line in range(len(label)):
        labels += label[line]
    print(labels)

def single_image():
    f = gzip.open("../data/t10k-images-idx3-ubyte.gz",'rb')
    label = f.readlines()
    labels = []
    for line in range(1):
        labels += label[line]
    difference = len(labels) - 784
    return labels[:len(labels) - difference]

def save_image(pixels:list):
    image = Image.open("../data/mnist_complete_zero.png")
    image = Image.new(image.mode,image.size)
    image.putdata(pixels)
    image.save("test.png")



a = single_image()
print(a)
save_image(a)
##to do
'''
- create memoize decorator
- figure out image
'''
