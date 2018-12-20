
import gzip
import struct
from PIL import Image

#global
E = 2.718281828459045

DATA_TYPES = {
    8: ('ubyte', 'B', 1),
    9: ('byte   ', 'b', 1),
    11: ('>i2', 'h', 2),
    12: ('>i4', 'i', 4),
    13: ('>f4', 'f', 4),
    15: ('>f8', 'd', 8)
    }

def sigmoid(x):
    "the sigmoid function"
    if x < (-700):
        return 0
    return 1.0/(1.0+(E**(-x)))


def sigmoid_prime(x):
    "derivative of the sigmoid function"
    "growth of the function"
    return sigmoid(x)*(1 - sigmoid(x))

def label_data():

    print("Training set labels")
    f = open("../data/t10k-labels.idx1-ubyte",'rb')
    f.seek(0)
    magic_num = struct.unpack('>4B',f.read(4))
    print("data type: {}".format(DATA_TYPES[magic_num[2]][0]))
    print("data format: {}".format(DATA_TYPES[magic_num[2]][1]))
    print("data type: {}".format(DATA_TYPES[magic_num[2]][2]))

    print()
    f.seek(4)
    num_labels = struct.unpack('>I',f.read(4))[0]
    print("labels:   {}".format(num_labels))
    print()
    labels = []
    for i in range(num_labels):
        labels.append(struct.unpack(">B",f.read(1))[0])
    return labels
    


    
    

def image_data():
    '''
    reads data from the MNIST data set 
    '''

    print("Training set images")
    f = open("../data/t10k-images.idx3-ubyte",'rb')
    f.seek(0)
    magic_num = struct.unpack('>4B',f.read(4))
    print("data type: {}".format(DATA_TYPES[magic_num[2]][0]))
    print("data format: {}".format(DATA_TYPES[magic_num[2]][1]))
    print("data type: {}".format(DATA_TYPES[magic_num[2]][2]))

    print()
    f.seek(4)
    num_images = struct.unpack('>I',f.read(4))[0]
    rows = struct.unpack('>I',f.read(4))[0]
    columns = struct.unpack('>I',f.read(4))[0]
    print("images: {}".format(num_images))
    print("rows:   {}".format(rows))
    print("columns:{}".format(columns))
    print()
    images = []
    for i in range(num_images):
        images.append(struct.unpack('>' + 'B'*(rows*columns)\
                                    ,f.read(rows*columns)))
    return images


def data_set():
    '''
    zips the set of image pixel data 
    along with the set of label test data 
    '''
    images = image_data()
    labels = label_data()
    return [i for i in zip(images, labels)]

def save_image(pixels:list):

    image = Image.new('L', (28,28))
    image.putdata(pixels)
    image.save("test.png")


a = data_set()
b = convert_data(a)
print(b[0])


##to do
'''
- create memoize decorator
'''
